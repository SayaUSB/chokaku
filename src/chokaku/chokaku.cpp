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

Inference::~Inference() {
    stop_realtime_classification();
    
    if (pa_stream_) {
        Pa_CloseStream(pa_stream_);
    }
    Pa_Terminate();
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

std::vector<float> Inference::resample_audio(const std::vector<float>& waveform,
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

std::vector<float> Inference::preprocess_audio(const std::vector<float>& waveform,
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
PredictionResult Inference::predict(const std::vector<float>& audio_data, 
                                                 int sample_rate) {
    auto processed = preprocess_audio(audio_data, sample_rate);
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
std::vector<float> Inference::capture_audio_blocking(float duration, 
                                                                  int sample_rate) {
    size_t num_samples = static_cast<size_t>(duration * sample_rate);
    std::vector<float> buffer(num_samples);
    
    PaStream* stream;
    
    // Set up input parameters
    PaStreamParameters inputParams;
    
    // Validate the specified device
    if (config_.mic_id >= 0 && config_.mic_id < Pa_GetDeviceCount()) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(config_.mic_id);
        if (deviceInfo->maxInputChannels > 0) {
            inputParams.device = config_.mic_id;
        } else {
            std::cerr << "Device " << config_.mic_id << " (" << deviceInfo->name 
                      << ") has no input channels, falling back to default input device\n";
            inputParams.device = Pa_GetDefaultInputDevice();
        }
    } else {
        std::cerr << "Invalid device ID " << config_.mic_id 
                  << ", falling back to default input device\n";
        inputParams.device = Pa_GetDefaultInputDevice();
    }
    
    // Check if we have a valid device
    if (inputParams.device == paNoDevice) {
        std::cerr << "No valid input device found\n";
        return {};
    }
    
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(inputParams.device);
    std::cout << "Using device: " << deviceInfo->name << " (max input channels: " 
              << deviceInfo->maxInputChannels << ")\n";
    
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = deviceInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;
    
    PaError err = Pa_OpenStream(&stream,
                              &inputParams,     // input parameters
                              nullptr,          // output parameters (none)
                              sample_rate,
                              256,             // frames per buffer
                              paClipOff,       // stream flags
                              nullptr,         // callback (blocking mode)
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

void Inference::capture_loop(float duration, int sample_rate) {
    while (running_.load()) {
        auto audio_data = capture_audio_blocking(duration, sample_rate);
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

int Inference::pa_callback(const void* input_buffer, void* output_buffer,
                                          unsigned long frames_per_buffer,
                                          const PaStreamCallbackTimeInfo* time_info,
                                          PaStreamCallbackFlags status_flags,
                                          void* user_data) {
    // Callback implementation for non-blocking mode if needed
    auto* self = static_cast<Inference*>(user_data);
    // ... handle streaming buffer
    return paContinue;
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

// ============================================================================
// Utility Functions
// ============================================================================



} // namespace chokaku