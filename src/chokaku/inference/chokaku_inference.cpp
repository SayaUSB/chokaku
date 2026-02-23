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

#include "chokaku/inference/chokaku_inference.hpp"
#include "chokaku/utils/csv.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>

namespace chokaku {

Inference::Inference(const ChokakuConfig& config)
    : config_(config) {
    
    // Build file paths from config
    std::string dir = config_.model_dir;
    if (!dir.empty() && dir.back() == '/') {
        dir.pop_back();
    }
    
    xml_path_ = dir + "/" + config_.model_xml;
    bin_path_ = dir + "/" + config_.model_bin;
    class_map_path_ = dir + "/" + config_.model_class_map;
    
    std::cout << "Loading model files:\n";
    std::cout << "XML: " << xml_path_ << "\n";
    std::cout << "BIN: " << bin_path_ << "\n";
    std::cout << "CSV: " << class_map_path_ << "\n";

    std::ifstream xml_file(xml_path_);
    if (!xml_file.good()) {
        throw std::runtime_error("XML model file not found: " + xml_path_);
    }
    xml_file.close();

    std::ifstream bin_file(bin_path_);
    if (!bin_file.good()) {
        throw std::runtime_error("Binary weights file not found: " + bin_path_ + 
                                "\nThe .bin file must reside in the same directory as the .xml file.");
    }
    bin_file.close();

    audio_capture_ = std::make_unique<AudioCapture>(config_);

    load_model();
    load_class_map();

    std::cout << "Chokaku OpenVINO inference initialized\n";
    std::cout << "Model XML : " << xml_path_ << "\n";
    std::cout << "Model BIN : " << bin_path_ << "\n";
}

Inference::~Inference() {
    stop_realtime_classification();
}

void Inference::load_model() {
    try {
        model_ = core_.read_model(xml_path_);
        
        auto inputs = model_->inputs();
        auto outputs = model_->outputs();
        
        if (inputs.empty() || outputs.empty()) {
            throw std::runtime_error("Model has no inputs or outputs");
        }

        auto input_node = inputs[0];
        auto output_node = outputs[output_idx_];
        output_name_ = output_node.get_any_name();
        input_name_ = input_node.get_any_name();
        
        // Reshape input to fixed size: 15360 samples (0.96s @ 16kHz)
        const int target_length = 15360;
        ov::Shape new_shape = {static_cast<size_t>(target_length)};
        model_->reshape({{input_name_, new_shape}});

        compiled_model_ = core_.compile_model(model_, "CPU");
        infer_request_ = compiled_model_.create_infer_request();
        input_shape_ = compiled_model_.input(input_name_).get_shape();
        output_shape_ = compiled_model_.output(output_idx_).get_shape();

    } catch (const std::exception& e) {
        std::cerr << "Error loading OpenVINO model: " << e.what() << "\n";
        throw;
    }
}

void Inference::load_class_map() {
    try {
        io::CSVReader<3> csv(class_map_path_);
        csv.read_header(io::ignore_extra_column, "index", "mid", "display_name");
        
        std::string index, mid, display_name;
        while (csv.read_row(index, mid, display_name)) {
            class_map_[index] = display_name;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading class map: " << e.what() << "\n";
    }
}

// Inference
PredictionResult Inference::predict(const std::vector<float>& audio_data, 
                                                 int sample_rate) {
    auto processed = preprocess_audio(audio_data, sample_rate, 16000, 0.96f);
    if (processed.empty()) {
        return {{}, {}, -1};
    }
    return predict(processed.data(), processed.size(), 16000);
}

PredictionResult Inference::predict(const float* audio_data, 
                                                 size_t length, 
                                                 int sample_rate) {
    PredictionResult result;
    result.top_class_index = -1;
    
    try {
        ov::Tensor input_tensor(compiled_model_.input(input_name_).get_element_type(), 
                               input_shape_);
        
        float* input_data = input_tensor.data<float>();
        size_t copy_len = std::min(length, input_shape_[0]);
        std::memcpy(input_data, audio_data, copy_len * sizeof(float));
        
        if (copy_len < input_shape_[0]) {
            std::fill(input_data + copy_len, input_data + input_shape_[0], 0.0f);
        }

        infer_request_.set_input_tensor(input_tensor);
        infer_request_.infer();

        ov::Tensor output_tensor = infer_request_.get_output_tensor(output_idx_);
        const float* predictions = output_tensor.data<const float>();
        
        auto out_shape = output_tensor.get_shape();
        size_t num_classes = 1;
        for (auto dim : out_shape) num_classes *= dim;
        
        if (out_shape.size() == 2 && out_shape[0] == 1) {
            num_classes = out_shape[1];
        }

        std::vector<std::pair<float, size_t>> scored_indices;
        for (size_t i = 0; i < num_classes; ++i) {
            scored_indices.push_back({predictions[i], i});
        }
        
        std::partial_sort(scored_indices.begin(), 
                         scored_indices.begin() + std::min(size_t(5), num_classes),
                         scored_indices.end(),
                         std::greater<std::pair<float, size_t>>());

        for (size_t i = 0; i < std::min(size_t(5), num_classes); ++i) {
            size_t idx = scored_indices[i].second;
            float score = scored_indices[i].first;
            std::string class_name = std::to_string(idx);
            
            auto it = class_map_.find(std::to_string(idx));
            if (it != class_map_.end()) {
                class_name = it->second;
            }
            
            result.predicted_classes.push_back(class_name);
            result.confidence_scores.push_back(score);
        }
        
        if (!result.predicted_classes.empty()) {
            result.top_class_index = static_cast<int>(scored_indices[0].second);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error during OpenVINO inference: " << e.what() << "\n";
        return {{}, {}, -1};
    }
    
    return result;
}

// Real-time Classification
void Inference::capture_loop(float duration, int sample_rate) {
    while (running_.load()) {
        auto audio_data = audio_capture_->capture_audio_blocking(duration, sample_rate);
        if (!audio_data.empty()) {
            auto result = predict(audio_data, sample_rate);
            print_prediction(result);
        }
    }
}

void Inference::start_realtime_classification(float duration, int sample_rate) {
    if (running_.load()) {
        std::cerr << "Real-time classification already running\n";
        return;
    }
    
    running_ = true;
    std::cout << "Starting OpenVINO real-time audio classification...\n";
    std::cout << "Press Ctrl+C to stop\n";
    
    // Run in blocking mode in main thread for simplicity, or spawn thread
    capture_loop(duration, sample_rate);
}

void Inference::stop_realtime_classification() {
    running_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
}

bool Inference::is_running() const {
    return running_.load();
}

void Inference::print_model_info() const {
    std::cout << "\n=== OpenVINO Model Information ===\n";
    std::cout << "XML path: " << xml_path_ << "\n";
    std::cout << "BIN path: " << bin_path_ << "\n";
    
    auto devices = core_.get_available_devices();
    std::cout << "Available devices: ";
    for (const auto& dev : devices) {
        std::cout << dev << " ";
    }
    std::cout << "\n";
    
    try {
        auto ops = model_->get_ops();
        std::cout << "Number of operations: " << ops.size() << "\n";
    } catch (...) {
        std::cout << "Number of operations: Unable to determine\n";
    }
    
    std::cout << std::string(40, '=') << "\n";
}

std::string Inference::get_input_name() const {
    return input_name_;
}

std::string Inference::get_output_name() const {
    return output_name_;
}

size_t Inference::get_num_classes() const {
    return class_map_.size();
}

} // namespace chokaku