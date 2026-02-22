#include "chokaku/chokaku.hpp"
#include "chokaku/utils/csv.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>

namespace chokaku {

ChokakuOpenVINOInference::ChokakuOpenVINOInference(const std::string& xml_path,
                                                 const std::string& class_map_path)
    : xml_path_(xml_path), class_map_path_(class_map_path) {
    
    size_t last_dot = xml_path_.find_last_of('.');
    if (last_dot != std::string::npos) {
        bin_path_ = xml_path_.substr(0, last_dot) + ".bin";
    } else {
        bin_path_ = xml_path_ + ".bin";
    }

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

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw std::runtime_error(std::string("PortAudio initialization failed: ") + Pa_GetErrorText(err));
    }

    load_model();
    load_class_map();

    std::cout << "Chokaku OpenVINO inference initialized\n";
    std::cout << "Model XML : " << xml_path_ << "\n";
    std::cout << "Model BIN : " << bin_path_ << "\n";
}

ChokakuOpenVINOInference::~ChokakuOpenVINOInference() {
    stop_realtime_classification();
    
    if (pa_stream_) {
        Pa_CloseStream(pa_stream_);
    }
    Pa_Terminate();
}

void ChokakuOpenVINOInference::load_model() {
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

void ChokakuOpenVINOInference::load_class_map() {
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

std::vector<float> ChokakuOpenVINOInference::resample_audio(const std::vector<float>& waveform,
                                                          int src_rate, int dst_rate) {
    if (src_rate == dst_rate) return waveform;
    
    double duration = static_cast<double>(waveform.size()) / src_rate;
    size_t target_length = static_cast<size_t>(duration * dst_rate);
    
    std::vector<float> resampled(target_length);
    for (size_t i = 0; i < target_length; ++i) {
        double src_idx = static_cast<double>(i) * src_rate / dst_rate;
        size_t idx_low = static_cast<size_t>(std::floor(src_idx));
        size_t idx_high = static_cast<size_t>(std::ceil(src_idx));
        double frac = src_idx - idx_low;
        
        if (idx_high >= waveform.size()) idx_high = waveform.size() - 1;
        
        resampled[i] = waveform[idx_low] * (1.0f - frac) + waveform[idx_high] * frac;
    }
    
    return resampled;
}

std::vector<float> ChokakuOpenVINOInference::preprocess_audio(const std::vector<float>& waveform,
                                                           int sample_rate,
                                                           int target_sample_rate,
                                                           float target_duration) {
    std::vector<float> mono = waveform;
    
    std::vector<float> resampled = resample_audio(mono, sample_rate, target_sample_rate);
    
    size_t target_length = static_cast<size_t>(target_sample_rate * target_duration);
    
    std::vector<float> processed;
    if (resampled.size() < target_length) {
        processed = resampled;
        processed.resize(target_length, 0.0f);
    } else if (resampled.size() > target_length) {
        processed.assign(resampled.begin(), resampled.begin() + target_length);
    } else {
        processed = resampled;
    }
    
    return processed;
}

// Inference
PredictionResult ChokakuOpenVINOInference::predict(const std::vector<float>& audio_data, 
                                                 int sample_rate) {
    auto processed = preprocess_audio(audio_data, sample_rate);
    if (processed.empty()) {
        return {{}, {}, -1};
    }
    return predict(processed.data(), processed.size(), 16000);
}

PredictionResult ChokakuOpenVINOInference::predict(const float* audio_data, 
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
std::vector<float> ChokakuOpenVINOInference::capture_audio_blocking(float duration, 
                                                                  int sample_rate) {
    size_t num_samples = static_cast<size_t>(duration * sample_rate);
    std::vector<float> buffer(num_samples);
    
    PaStream* stream;
    PaError err = Pa_OpenDefaultStream(&stream,
                                       1,          // input channels (mono)
                                       0,          // output channels
                                       paFloat32,  // sample format
                                       sample_rate,
                                       256,        // frames per buffer
                                       nullptr,    // callback (blocking mode)
                                       nullptr);
    
    if (err != paNoError) {
        std::cerr << "Failed to open PortAudio stream: " << Pa_GetErrorText(err) << "\n";
        return {};
    }
    
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Failed to start stream: " << Pa_GetErrorText(err) << "\n";
        Pa_CloseStream(stream);
        return {};
    }
    
    // Read audio
    err = Pa_ReadStream(stream, buffer.data(), num_samples);
    if (err != paNoError && err != paInputOverflowed) {
        std::cerr << "Error reading audio: " << Pa_GetErrorText(err) << "\n";
    }
    
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    
    return buffer;
}

void ChokakuOpenVINOInference::capture_loop(float duration, int sample_rate) {
    while (running_.load()) {
        auto audio_data = capture_audio_blocking(duration, sample_rate);
        if (!audio_data.empty()) {
            auto result = predict(audio_data, sample_rate);
            print_prediction(result);
        }
    }
}

void ChokakuOpenVINOInference::start_realtime_classification(float duration, int sample_rate) {
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

void ChokakuOpenVINOInference::stop_realtime_classification() {
    running_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
}

bool ChokakuOpenVINOInference::is_running() const {
    return running_.load();
}

int ChokakuOpenVINOInference::pa_callback(const void* input_buffer, void* output_buffer,
                                          unsigned long frames_per_buffer,
                                          const PaStreamCallbackTimeInfo* time_info,
                                          PaStreamCallbackFlags status_flags,
                                          void* user_data) {
    // Callback implementation for non-blocking mode if needed
    auto* self = static_cast<ChokakuOpenVINOInference*>(user_data);
    // ... handle streaming buffer
    return paContinue;
}

void ChokakuOpenVINOInference::print_model_info() const {
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

std::string ChokakuOpenVINOInference::get_input_name() const {
    return input_name_;
}

std::string ChokakuOpenVINOInference::get_output_name() const {
    return output_name_;
}

size_t ChokakuOpenVINOInference::get_num_classes() const {
    return class_map_.size();
}

// ============================================================================
// Utility Functions
// ============================================================================

std::map<std::string, std::string> load_csv_class_map(const std::string& csv_path) {
    std::map<std::string, std::string> class_map;
    std::ifstream file(csv_path);
    
    if (!file.is_open()) {
        std::cerr << "Could not open class map: " << csv_path << "\n";
        return class_map;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string index, mid, display_name;
        
        if (std::getline(ss, index, ',') && 
            std::getline(ss, mid, ',') && 
            std::getline(ss, display_name)) {
            
            auto trim = [](std::string& s) {
                if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
                    s = s.substr(1, s.size() - 2);
                }
            };
            trim(index);
            trim(display_name);
            class_map[index] = display_name;
        }
    }
    
    return class_map;
}

void print_prediction(const PredictionResult& result, size_t top_k) {
    if (result.predicted_classes.empty()) {
        std::cout << "Failed to classify audio.\n";
        return;
    }
    
    size_t num_to_print = std::min(top_k, result.predicted_classes.size());
    for (size_t i = 0; i < num_to_print; ++i) {
        std::cout << "  " << (i + 1) << ". " 
                  << result.predicted_classes[i] << ": "
                  << std::fixed << std::setprecision(4) 
                  << result.confidence_scores[i] << "\n";
    }
}

} // namespace chokaku