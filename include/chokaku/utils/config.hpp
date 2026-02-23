#ifndef CHOKAKU_CONFIG_HPP
#define CHOKAKU_CONFIG_HPP

#include <string>
#include <fstream>
#include <sstream>

namespace chokaku {

struct ChokakuConfig {
    int mic_id = 0;
    std::string model_dir = "./models";
    std::string model_xml = "chokaku.xml";
    std::string model_bin = "chokaku.bin";
    std::string model_class_map = "chokaku_class_map.csv";
    int sample_rate = 16000;
    float chunk_duration = 0.96f;
    int audio_buffer_size = 512;
    
    static ChokakuConfig load_from_json(const std::string& config_path);
};

} // namespace chokaku

#endif // CHOKAKU_CONFIG_HPP
