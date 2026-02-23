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
#include <memory>
#include "chokaku/utils/config.hpp"
#include "chokaku/utils/prediction.hpp"
#include "chokaku/utils/audio_processing.hpp"
#include "chokaku/utils/audio_capture.hpp"

namespace chokaku {

class Inference {
public:
    Inference(const ChokakuConfig& config);
    ~Inference();

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;
    Inference(Inference&&) = delete;
    Inference& operator=(Inference&&) = delete;

    PredictionResult predict(const std::vector<float>& audio_data, int sample_rate = 16000);
    PredictionResult predict(const float* audio_data, size_t length, int sample_rate = 16000);
    
    void start_realtime_classification(float duration = 1.0f, int sample_rate = 16000);
    void stop_realtime_classification();
    bool is_running() const;

    void print_model_info() const;
    std::string get_input_name() const;
    std::string get_output_name() const;
    size_t get_num_classes() const;

private:
    ChokakuConfig config_;
    
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

    // Audio capture
    std::unique_ptr<AudioCapture> audio_capture_;

    // Private methods
    void load_model();
    void load_class_map();
    
    void capture_loop(float duration, int sample_rate);
};

} // namespace chokaku

#endif // CHOKAKU_OPENVINO_HPP