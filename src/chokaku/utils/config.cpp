#include "chokaku/utils/config.hpp"
#include <iostream>

namespace chokaku {

ChokakuConfig ChokakuConfig::load_from_json(const std::string& config_path) {
    ChokakuConfig config;
    
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Could not open config file: " << config_path << ", using defaults.\n";
        return config;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        
        if (line.empty() || line[0] == '{' || line[0] == '}' || line[0] == ',') {
            continue;
        }
        
        // Parse key-value pairs
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            
            // Clean up key and value
            key.erase(0, key.find_first_not_of(" \t\""));
            key.erase(key.find_last_not_of(" \t\"") + 1);
            value.erase(0, value.find_first_not_of(" \t,"));
            value.erase(value.find_last_not_of(" \t,") + 1);
            
            // Remove quotes from value if present
            if (!value.empty() && value.front() == '"') {
                value.erase(0, 1);
            }
            if (!value.empty() && value.back() == '"') {
                value.pop_back();
            }
            
            if (key == "mic_id") {
                config.mic_id = std::stoi(value);
            } else if (key == "model_dir") {
                config.model_dir = value;
            } else if (key == "model_xml") {
                config.model_xml = value;
            } else if (key == "model_bin") {
                config.model_bin = value;
            } else if (key == "model_class_map") {
                config.model_class_map = value;
            } else if (key == "sample_rate") {
                config.sample_rate = std::stoi(value);
            } else if (key == "chunk_duration") {
                config.chunk_duration = std::stof(value);
            } else if (key == "audio_buffer_size") {
                config.audio_buffer_size = std::stoi(value);
            }
        }
    }
    
    return config;
}

} // namespace chokaku
