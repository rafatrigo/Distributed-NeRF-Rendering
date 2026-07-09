#ifndef METRICS_LOGGER_H
#define METRICS_LOGGER_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <sys/stat.h>

class MetricsLogger {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    std::string csv_filename;

    // Função auxiliar para checar se o arquivo existe
    bool fileExists(const std::string& name) {
        struct stat buffer;   
        return (stat(name.c_str(), &buffer) == 0); 
    }

public:
    MetricsLogger(const std::string& filename = "nerf_execution_metrics.csv") : csv_filename(filename) {
        if (!fileExists(csv_filename)) {
            std::ofstream file(csv_filename);
            if (file.is_open()) {
                // Cabeçalho do CSV
                file << "timestamp,num_workers,image_width,image_height,execution_time_ms\n";
                file.close();
            } else {
                std::cerr << "[MetricsLogger] Erro ao criar o arquivo CSV: " << csv_filename << "\n";
            }
        }
    }

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    void save(int num_workers, int width, int height) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Captura o tempo atual para o log
        auto now = std::chrono::system_clock::now();
        std::time_t time_t_now = std::chrono::system_clock::to_time_t(now);
        
        // Remove a quebra de linha padrão do ctime
        std::string time_str = std::ctime(&time_t_now);
        time_str.pop_back();

        std::ofstream file(csv_filename, std::ios::app);
        if (file.is_open()) {
            file << "\"" << time_str << "\","
                 << num_workers << ","
                 << width << ","
                 << height << ","
                 << duration << "\n";
            file.close();
            std::cout << "[MetricsLogger] Métricas salvas! " 
                      << num_workers << " workers finalizaram em " 
                      << duration << " ms.\n";
        } else {
            std::cerr << "[MetricsLogger] Erro ao abrir o CSV para escrita!\n";
        }
    }
};

#endif // METRICS_LOGGER_H