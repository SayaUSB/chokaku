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

#ifndef CHOKAKU_NODE_HPP
#define CHOKAKU_NODE_HPP

#include "chokaku/inference/chokaku_inference.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include <memory>
#include <chrono>

class ChokakuNode : public rclcpp::Node {
public:
    ChokakuNode();
    ~ChokakuNode();

private:
    void check_whistle_detection();
    
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr whistle_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<chokaku::Inference> inference_;
    chokaku::ChokakuConfig config_;
};

#endif // CHOKAKU_NODE_HPP
