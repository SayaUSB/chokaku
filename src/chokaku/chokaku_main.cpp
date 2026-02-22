#include "chokaku/chokaku.hpp"
#include <iostream>
#include <csignal>
#include <cstring>

namespace {
    chokaku::ChokakuOpenVINOInference* g_inference = nullptr;
    
    void signal_handler(int sig) {
        std::cout << "\nStopping OpenVINO classification...\n";
        if (g_inference) {
            g_inference->stop_realtime_classification();
        }
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --config <path>      Path to JSON config file (default: config/chokaku_config.json)\n"
              << "  --info               Show model information and exit\n"
              << "  -h, --help           Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string config_path = "config/chokaku_config.json";
    bool show_info = false;
    
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--info") == 0) {
            show_info = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    try {
        chokaku::ChokakuConfig config = chokaku::ChokakuConfig::load_from_json(config_path);
        chokaku::ChokakuOpenVINOInference inference(config);
        g_inference = &inference;
        
        std::signal(SIGINT, signal_handler);
        
        if (show_info) {
            inference.print_model_info();
            return 0;
        }
        
        inference.start_realtime_classification(config.chunk_duration, config.sample_rate);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}