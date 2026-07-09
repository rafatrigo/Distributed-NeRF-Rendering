#include <iostream>
#include <zmq.hpp>
#include <vector>
#include <array>
#include <queue>
#include <string>
#include <filesystem>
#include <chrono>
#include <unordered_map>

#include "flatbuffers/flatbuffers.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "nerf_messages_generated.h"
#include "camera_math.h"
#include "metrics_logger.hpp"

struct RayTask {
  int batch_id;
  int start_idx;
  int end_idx;
  std::array<float, 16> pose_data; //keeps the data contiguous and on the stack
};

struct InFlightTask {
  RayTask task;
  std::chrono::steady_clock::time_point start_time;
};

//TODO: A static master is a unique fail point, maybe use an algorithm to chose a master dinamicaly
//TODO: pass things as a parameter or config file
int main() {
  std::cout << "Starting master node... (ROUTER)" << std::endl;

  MetricsLogger logger("nerf_speedup_metrics.csv");

  // ZeroMQ config (socket ROUTER)
  zmq::context_t context(1);
  zmq::socket_t broker(context, zmq::socket_type::router);
  broker.bind("tcp://*:5555");

  int loop_count = 0;
  while(loop_count < 10) {
    logger.start();

    const int TIMEOUT_SECONDS = 15;
    
    // camera parameters
    int H = 100;
    int W = 100;
    float focal = 138.88f;
    int total_rays = H*W;
    float near = 2.0f;
    float far = 6.0f;
    int n_samples = 64;
    
    int chunk_size = 2500;
    int total_frames = 120;
    int chunks_per_frame = (total_rays + chunk_size - 1) / chunk_size;
    int total_tasks = total_frames * chunks_per_frame;
    
    // images buffer | 120 frames -- 100x100 -- ~14.MB
    std::vector<std::vector<float>> final_images(total_frames, std::vector<float>(total_rays * 3, 0.0f));
    
    std::queue<RayTask> task_queue;
    
    int global_batch_counter = 0;
    
    for (int f = 0; f < total_frames; ++f) {
      float th = (360.0f / total_frames) * f;
      
      Matrix4fRM c2w = CameraMath::pose_spherical(th, -30.0f, 4.0f);
      std::array<float, 16> pose_data;
      std::copy(c2w.data(), c2w.data() + 16, pose_data.begin());
    
      for(int i = 0; i < total_rays; i += chunk_size) {
          int end = std::min(i + chunk_size, total_rays);
          task_queue.push({global_batch_counter++, i, end, pose_data});
      }
    }
    
    int completed_tasks = 0;
    
    std::queue<std::string> available_workers;

    std::unordered_map<std::string, InFlightTask> in_flight_tasks;
    
    std::cout << "waiting for workers... (" << total_tasks << " tasks on queue)\n" << std::endl;
    
    // main event loop
    while (completed_tasks < total_tasks) {

      /**
        Array of items to monitor
        - zeroMQ socket
        - file descriptot (if it was a OS system)
        - flag: ZMQ_POLLIN = when receive data
        - returned events - init with 0 zmq::poll will fill it
      */
      zmq::pollitem_t items[] = { { static_cast<void*>(broker), 0, ZMQ_POLLIN, 0 } };

      /**
        - Monitor the first item for at most 1000ms
        - If it don't receive anything sleep for 1000ms
      */
      zmq::poll(&items[0], 1, std::chrono::milliseconds(1000));

      // if returned events contain a flag ZMQ_POLLIN
      if (items[0].revents & ZMQ_POLLIN) {
        zmq::message_t identity;
        zmq::message_t empty;
        zmq::message_t payload;
      
        // the ROUTER always receive 3 frames of a REQ: [identity] - [empty] - [data]
        // dontwait flag = try to read, but if it's empty don't wait
        (void)broker.recv(identity, zmq::recv_flags::dontwait); //TODO: look in this cast
        (void)broker.recv(empty, zmq::recv_flags::dontwait);
        (void)broker.recv(payload, zmq::recv_flags::dontwait);
      
        std::string worker_id(static_cast<char*>(identity.data()), identity.size());
      
        // verify if the signal is "READY", "ERROR" or a FlatBuffer
        std::string payload_str(static_cast<char*>(payload.data()), payload.size());
      
        if (payload_str == "READY") {
          std::cout << "[+] worker connected/free: " << worker_id << std::endl;
          available_workers.push(worker_id);
        } else if (payload_str == "ERROR") {
            std::cerr << "[!] worker error: " << worker_id << std::endl;
            
            if (in_flight_tasks.count(worker_id)) {
                task_queue.push(in_flight_tasks[worker_id].task);
                in_flight_tasks.erase(worker_id);
            }
            available_workers.push(worker_id);
        } else {
          auto render_result = flatbuffers::GetRoot<NerfDistributed::RenderResult>(payload.data());
      
          int g_id = render_result->batch_id();
          auto rgb_map = render_result->rgb_map();
      
          // find which frame and image part the data belongs to
          int frame_id = g_id / chunks_per_frame;
          int local_b_id = g_id % chunks_per_frame;
      
          int start_offset = (local_b_id * chunk_size) * 3;
      
          for(size_t i = 0; i < rgb_map->size(); ++i) {
            final_images[frame_id][start_offset + i] = rgb_map->Get(i);
          }
      
          completed_tasks++;
      
          std::cout << "[-] batch: " << g_id << " concluded by worker: " << worker_id << " (" << completed_tasks << "/" << total_tasks << ")" << std::endl;
      
          in_flight_tasks.erase(worker_id);
          // the worker that replied is now free
          available_workers.push(worker_id);
        }
      }

      // TIMEOUT verification
      auto now = std::chrono::steady_clock::now();
      for (auto it = in_flight_tasks.begin(); it != in_flight_tasks.end(); ) {
          auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - it->second.start_time).count();
          
          if (duration > TIMEOUT_SECONDS) {
              std::cerr << "[!] Worker timeout: " << it->first << ". Re-queuing batch " 
                        << it->second.task.batch_id << std::endl;
              
              task_queue.push(it->second.task);
              
              it = in_flight_tasks.erase(it); 
          } else {
              ++it;
          }
      }
    
      // task dispatch: free workers AND pending tasks
      while (!available_workers.empty() && !task_queue.empty()) {
        std::string next_worker = available_workers.front();
        available_workers.pop();
    
        RayTask task = task_queue.front();
        task_queue.pop();

        // mark as doing
        in_flight_tasks[next_worker] = {task, std::chrono::steady_clock::now()};
    
        // build flatbuffer message
        flatbuffers::FlatBufferBuilder builder(1024);
        auto pose_vec = builder.CreateVector(task.pose_data.data(), task.pose_data.size());
        auto ray_batch = NerfDistributed::CreateRayBatch(
          builder, task.batch_id, task.start_idx, task.end_idx,
          H, W, focal, pose_vec, near, far, n_samples
        );
        builder.Finish(ray_batch);
    
        // the ROUTER must send 3 frames back: [identity] - [empty] - [data]
        zmq::message_t req_identity(next_worker.data(),  next_worker.size());
        zmq::message_t req_empty(0);
        zmq::message_t req_payload(builder.GetBufferPointer(), builder.GetSize());
    
        broker.send(req_identity, zmq::send_flags::sndmore);
        broker.send(req_empty, zmq::send_flags::sndmore);
        broker.send(req_payload, zmq::send_flags::none);
      }
    }
    
    logger.stop();
    logger.save(1, W, H);

    // poison pill loop
    while (!available_workers.empty()) {
      std::string worker_id = available_workers.front();
      available_workers.pop();
  
      flatbuffers::FlatBufferBuilder builder(256);
      // batch_id = -1 ends the worker
      auto poison_pill = NerfDistributed::CreateRayBatch(
        builder, -1, 0, 0, 0, 0, 0.0f, 0, 0.0f, 0.0f, 0
      );
  
      builder.Finish(poison_pill);
  
      zmq::message_t req_identity(worker_id.data(), worker_id.size());
      zmq::message_t req_empty(0);
      zmq::message_t req_payload(builder.GetBufferPointer(), builder.GetSize());
  
      broker.send(req_identity, zmq::send_flags::sndmore);
      broker.send(req_empty, zmq::send_flags::sndmore);
      broker.send(req_payload, zmq::send_flags::none);
    }
  }

  // std::cout << "rendering completed! finishing workers..." << std::endl;


  //----------- SAVE IMAGES --------------------------

  // std::cout << "\nSaving frames..." << std::endl;

  // std::filesystem::create_directory("video_frames");

  // for (int f = 0; f < total_frames; ++f) {
  //   std::vector<uint8_t> image_pixels(total_rays * 3);
    
  //   for (int i = 0; i < total_rays * 3; ++i) {
  //       float color_val = final_images[f][i];
  //       color_val = std::max(0.0f, std::min(1.0f, color_val)); 
  //       image_pixels[i] = static_cast<uint8_t>(color_val * 255.0f);
  //   }

  //   char filename[256];
  //   snprintf(filename, sizeof(filename), "video_frames/render_output_%03d.png", f);
    
  //   int stride_in_bytes = W * 3;
  //   int success = stbi_write_png(filename, W, H, 3, image_pixels.data(), stride_in_bytes);

  //   if (success) {
  //       std::cout << "Saved: " << f + 1 << "|" << total_frames << std::endl;
  //   } else {
  //       std::cerr << "[!] Fail to save image: " << filename << std::endl;
  //   }
  // }

  // std::cout << "\nAll frames saved." << std::endl;

  return 0;
}
