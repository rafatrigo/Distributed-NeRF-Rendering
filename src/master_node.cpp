#include <iostream>
#include <zmq.hpp>
#include <vector>
#include <queue>
#include <string>

#include "flatbuffers/flatbuffers.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "nerf_messages_generated.h"
#include "camera_math.h"

struct RayTask {
  int batch_id;
  int start_idx;
  int end_idx;
};

//TODO: A static master is a unique fail point, maybe use an algorithm to chose a master dinamicaly
//TODO: pass things as a parameter or config file
int main() {
  std::cout << "Starting master node... (ROUTER)" << std::endl;

  // ZeroMQ config (socket ROUTER)
  zmq::context_t context(1);
  zmq::socket_t broker(context, zmq::socket_type::router);
  broker.bind("tcp://*:5555");

  // camera parameters
  int H = 100;
  int W = 100;
  float focal = 138.88f;
  int total_rays = H*W;
  float near = 2.0f;
  float far = 6.0f;
  int n_samples = 64;

  Matrix4fRM c2w = CameraMath::pose_spherical(45.0f, -30.0f, 4.0f);
  std::vector<float> pose_data(c2w.data(), c2w.data() + 16);

  // final image buffer
  std::vector<float> final_image(total_rays * 3, 0.0f);

  std::queue<RayTask> task_queue;
  
  int chunk_size = 2500;
  int batch_counter = 0;

  for(int i = 0; i < total_rays; i += chunk_size) {
    int end  = std::min(i + chunk_size, total_rays);
    task_queue.push({batch_counter++, i, end});
  }

  int total_tasks = task_queue.size();
  int completed_tasks = 0;

  std::queue<std::string> available_workers;

  std::cout << "waiting for workers... (" << total_tasks << " tasks on queue)\n" << std::endl;

  // main event loop
  while (completed_tasks < total_tasks) {
    zmq::message_t identity;
    zmq::message_t empty;
    zmq::message_t payload;

    // the ROUTER always receive 3 frames of a REQ: [identity] - [empty] - [data]
    (void)broker.recv(identity, zmq::recv_flags::none); //TODO: look in this cast
    (void)broker.recv(empty, zmq::recv_flags::none);
    (void)broker.recv(payload, zmq::recv_flags::none);

    std::string worker_id(static_cast<char*>(identity.data()), identity.size());

    // verify if the signal is "READY", "ERROR" or a FlatBuffer
    std::string payload_str(static_cast<char*>(payload.data()), payload.size());

    if (payload_str == "READY" || payload_str == "ERROR") {
      std::cout << "[+] worker connected/free: " << worker_id << std::endl;
      available_workers.push(worker_id);
    } else {
      auto render_result = flatbuffers::GetRoot<NerfDistributed::RenderResult>(payload.data());

      int b_id = render_result->batch_id();
      auto rgb_map = render_result->rgb_map();

      int start_offset = (b_id * chunk_size) * 3;

      // image reconstruction in the final matrix
      for(size_t i = 0; i < rgb_map->size(); ++i) {
        final_image[start_offset + i] = rgb_map->Get(i);
      }

      completed_tasks++;

      std::cout << "[-] batch: " << b_id << " concluded by worker: " << worker_id << " (" << completed_tasks << "/" << total_tasks << ")" << std::endl;

      // the worker that replied is now free
      available_workers.push(worker_id);
    }

    // task dispatch: free workers AND pending tasks
    while (!available_workers.empty() && !task_queue.empty()) {
      std::string next_worker = available_workers.front();
      available_workers.pop();

      RayTask task = task_queue.front();
      task_queue.pop();

      // build flatbuffer message
      flatbuffers::FlatBufferBuilder builder(1024);
      auto pose_vec = builder.CreateVector(pose_data);
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

  std::cout << "rendering completed! finishing workers..." << std::endl;

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

  //----------- SAVE IMAGE --------------------------
  std::cout << "\nConvertendo matriz de cores (Float32 -> UInt8)..." << std::endl;

  // Cria um vetor para armazenar os pixels finais em 8 bits (RGB)
  std::vector<uint8_t> image_pixels(total_rays * 3);

  for (int i = 0; i < total_rays * 3; ++i) {
      // Replica o funcionamento do np.clip(rgb_map, 0, 1)
      float color_val = final_image[i];
      color_val = std::max(0.0f, std::min(1.0f, color_val)); 
      
      // Converte para escala de 0 a 255
      image_pixels[i] = static_cast<uint8_t>(color_val * 255.0f);
  }

  std::cout << "Salvando render_output.png no disco..." << std::endl;

  // Escreve a imagem em formato PNG
  // Parâmetros: nome do arquivo, largura, altura, canais de cor (3 para RGB), ponteiro dos dados, tamanho da linha em bytes (stride)
  int stride_in_bytes = W * 3;
  int success = stbi_write_png("render_output.png", W, H, 3, image_pixels.data(), stride_in_bytes);

  if (success) {
      std::cout << "Imagem salva com sucesso!" << std::endl;
  } else {
      std::cerr << "[!] Falha ao salvar a imagem." << std::endl;
  }

  return 0;
}
