#ifndef CHOKAKU_OPENVINO_HPP
#define CHOKAKU_OPENVINO_HPP

#include <openvino/openvino.hpp>
#include <portaudio.h>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>

namespace chokaku {

struct PredictionResult {
    std::vector<std::string> predicted_classes;
    std::vector<float> confidence_scores;
    int top_class_index;
};

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

class ChokakuOpenVINOInference {
public:
    ChokakuOpenVINOInference(const ChokakuConfig& config);
    ~ChokakuOpenVINOInference();

    ChokakuOpenVINOInference(const ChokakuOpenVINOInference&) = delete;
    ChokakuOpenVINOInference& operator=(const ChokakuOpenVINOInference&) = delete;
    ChokakuOpenVINOInference(ChokakuOpenVINOInference&&) = delete;
    ChokakuOpenVINOInference& operator=(ChokakuOpenVINOInference&&) = delete;

    PredictionResult predict(const std::vector<float>& audio_data, int sample_rate = 16000);
    PredictionResult predict(const float* audio_data, size_t length, int sample_rate = 16000);
    
    void start_realtime_classification(float duration = 1.0f, int sample_rate = 16000);
    void stop_realtime_classification();
    bool is_running() const;

    void print_model_info() const;
    std::string get_input_name() const;
    std::string get_output_name() const;
    size_t get_num_classes() const;

    static std::vector<float> preprocess_audio(const std::vector<float>& waveform, 
                                               int sample_rate = 16000,
                                               int target_sample_rate = 16000,
                                               float target_duration = 0.96f);

private:
    // Configuration
    ChokakuConfig config_;
    
    // File paths
    std::string xml_path_;
    std::string bin_path_;
    std::string class_map_path_;

    // OpenVINO objects
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    size_t output_idx_ = 0;


    // Model metadata
    std::string input_name_;
    std::string output_name_;
    ov::Shape input_shape_;
    ov::Shape output_shape_;

    // Class mapping
    std::map<std::string, std::string> class_map_;

    // Real-time threading
    std::atomic<bool> running_{false};
    std::thread capture_thread_;
    mutable std::mutex mutex_;

    // PortAudio
    PaStream* pa_stream_{nullptr};
    std::vector<float> audio_buffer_;
    std::mutex buffer_mutex_;

    // Private methods
    void load_model();
    void load_class_map();
    static std::vector<float> resample_audio(const std::vector<float>& waveform, 
                                             int src_rate, int dst_rate);
    
    // PortAudio callback
    static int pa_callback(const void* input_buffer, void* output_buffer,
                          unsigned long frames_per_buffer,
                          const PaStreamCallbackTimeInfo* time_info,
                          PaStreamCallbackFlags status_flags,
                          void* user_data);
    
    void capture_loop(float duration, int sample_rate);
    std::vector<float> capture_audio_blocking(float duration, int sample_rate);
};

// Utility functions
std::map<std::string, std::string> load_csv_class_map(const std::string& csv_path);
void print_prediction(const PredictionResult& result, size_t top_k = 3);

} // namespace chokaku

#endif // CHOKAKU_OPENVINO_HPP