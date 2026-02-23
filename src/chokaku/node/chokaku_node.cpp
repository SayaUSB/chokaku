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

#include "chokaku/node/chokaku_node.hpp"
#include <csignal>
#include <string>

namespace {
    chokaku::Inference* g_inference = nullptr;
    
    void signal_handler(int sig) {
        std::cout << "\nStopping OpenVINO classification...\n";
        if (g_inference) {
            g_inference->stop_realtime_classification();
        }
    }
}

ChokakuNode::ChokakuNode() : Node("chokaku_node") {
    // Initialize publisher
    whistle_publisher_ = this->create_publisher<std_msgs::msg::Bool>("chokaku/detection", 10);
    
    // Get config path
    std::string config_path = "config/chokaku_config.json";
    this->declare_parameter("config_path", config_path);
    config_path = this->get_parameter("config_path").as_string();
    
    RCLCPP_INFO(this->get_logger(), "Starting Chokaku whistle detection node");
    RCLCPP_INFO(this->get_logger(), "Config path: %s", config_path.c_str());
    
    try {
        // Load configuration and initialize inference
        config_ = chokaku::ChokakuConfig::load_from_json(config_path);
        inference_ = std::make_unique<chokaku::Inference>(config_);
        g_inference = inference_.get();
        
        // Start real-time classification
        inference_->start_realtime_classification(config_.chunk_duration, config_.sample_rate);
        
        RCLCPP_INFO(this->get_logger(), "Whistle detection started successfully");
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error initializing inference: %s", e.what());
        rclcpp::shutdown();
        return;
    }
    
    // Create timer for checking predictions
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&ChokakuNode::check_whistle_detection, this)
    );
}

ChokakuNode::~ChokakuNode() {
    if (inference_) {
        inference_->stop_realtime_classification();
    }
}

void ChokakuNode::check_whistle_detection() {
    if (!inference_ || !inference_->is_running()) {
        return;
    }
    
    // Get the latest prediction result
    auto prediction = inference_->get_latest_prediction();
    
    // Check if the top prediction is "Whistling" (index 35) or "Whistle" (index 396)
    bool is_whistle = false;
    if (prediction.top_class_index == 35 || prediction.top_class_index == 396) {
        is_whistle = true;
    }
    
    static bool last_state = false;
    bool current_state = is_whistle;
    
    // Publish and log state changes
    if (current_state != last_state) {
        auto message = std_msgs::msg::Bool();
        message.data = current_state;
        whistle_publisher_->publish(message);
        
        if (current_state) {
            RCLCPP_INFO(this->get_logger(), "Whistle detected! Class: %s (index: %d)", 
                       prediction.predicted_classes.empty() ? "Unknown" : prediction.predicted_classes[0].c_str(),
                       prediction.top_class_index);
        } else {
            RCLCPP_INFO(this->get_logger(), "Whistle detection ended");
        }
        last_state = current_state;
    }
}
