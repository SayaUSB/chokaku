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
              << "  --xml <path>         Path to OpenVINO IR XML model (default: chokaku.xml)\n"
              << "  --class-map <path>   Path to class map CSV (default: chokaku_class_map.csv)\n"
              << "  --info               Show model information and exit\n"
              << "  --duration <sec>     Audio chunk duration in seconds (default: 1.0)\n"
              << "  --sample-rate <hz>   Sample rate for recording (default: 16000)\n"
              << "  -h, --help           Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default arguments
    std::string xml_path = "chokaku.xml";
    std::string class_map_path = "chokaku_class_map.csv";
    bool show_info = false;
    float duration = 1.0f;
    int sample_rate = 16000;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--xml") == 0 && i + 1 < argc) {
            xml_path = argv[++i];
        } else if (std::strcmp(argv[i], "--class-map") == 0 && i + 1 < argc) {
            class_map_path = argv[++i];
        } else if (std::strcmp(argv[i], "--info") == 0) {
            show_info = true;
        } else if (std::strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) {
            sample_rate = std::stoi(argv[++i]);
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
        chokaku::ChokakuOpenVINOInference inference(xml_path, class_map_path);
        g_inference = &inference;
        
        // Setup signal handler for graceful shutdown
        std::signal(SIGINT, signal_handler);
        
        if (show_info) {
            inference.print_model_info();
            return 0;
        }
        
        // Start real-time classification
        inference.start_realtime_classification(duration, sample_rate);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}