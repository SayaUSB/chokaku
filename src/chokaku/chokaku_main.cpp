#include "chokaku/chokaku.hpp"
#include <iostream>
#include <csignal>

namespace {
    chokaku::Inference* g_inference = nullptr;
    
    void signal_handler(int sig) {
        std::cout << "\nStopping OpenVINO classification...\n";
        if (g_inference) {
            g_inference->stop_realtime_classification();
        }
    }
}

int main() {
    const std::string config_path = "config/chokaku_config.json";
    
    try {
        chokaku::ChokakuConfig config = chokaku::ChokakuConfig::load_from_json(config_path);
        chokaku::Inference inference(config);
        g_inference = &inference;
        
        std::signal(SIGINT, signal_handler);
        
        inference.start_realtime_classification(config.chunk_duration, config.sample_rate);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}