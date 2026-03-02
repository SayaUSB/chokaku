// MIT License

// Copyright (c) 2026 ICHIRO ITS

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "chokaku/utils/config.hpp"

#include <filesystem>
#include <stdexcept>

#include "jitsuyo/jitsuyo.hpp"
#include <nlohmann/json.hpp>

namespace chokaku {

using json = nlohmann::json;

ChokakuConfig ChokakuConfig::load_from_json(const std::string& config_path) {
    try {
        json config;
        std::filesystem::path p(config_path);
        std::string config_dir = p.parent_path().string() + "/";
        std::string config_file = p.filename().string();
        
        if (!jitsuyo::load_config(config_dir, config_file, config)) {
            throw std::runtime_error("Failed to load configuration file: " + config_path);
        }
        
        ChokakuConfig cfg;
        bool valid_config = true;
        valid_config &= jitsuyo::assign_val(config, "mic_id", cfg.mic_id);
        valid_config &= jitsuyo::assign_val(config, "model_dir", cfg.model_dir);
        valid_config &= jitsuyo::assign_val(config, "model_xml", cfg.model_xml);
        valid_config &= jitsuyo::assign_val(config, "model_bin", cfg.model_bin);
        valid_config &= jitsuyo::assign_val(config, "model_class_map", cfg.model_class_map);
        valid_config &= jitsuyo::assign_val(config, "sample_rate", cfg.sample_rate);
        valid_config &= jitsuyo::assign_val(config, "chunk_duration", cfg.chunk_duration);
        valid_config &= jitsuyo::assign_val(config, "audio_buffer_size", cfg.audio_buffer_size);
        valid_config &= jitsuyo::assign_val(config, "confidence_threshold", cfg.confidence_threshold);
        valid_config &= jitsuyo::assign_val(config, "top_predictions_output", cfg.top_predictions_output);
        valid_config &= jitsuyo::assign_val(config, "print_output", cfg.print_output);
        valid_config &= jitsuyo::assign_val(config, "enable_preprocessing", cfg.enable_preprocessing);
        valid_config &= jitsuyo::assign_val(config, "bandpass_low_freq", cfg.bandpass_low_freq);
        valid_config &= jitsuyo::assign_val(config, "bandpass_high_freq", cfg.bandpass_high_freq);
        valid_config &= jitsuyo::assign_val(config, "preemphasis_alpha", cfg.preemphasis_alpha);
        valid_config &= jitsuyo::assign_val(config, "noise_gate_threshold", cfg.noise_gate_threshold);
        
        if (!valid_config) {
            throw std::runtime_error("Invalid config at " + config_path);
        }
        
        return cfg;
    } catch (const std::exception& e) {
        throw;
    }
}

} // namespace chokaku
