#include "chokaku/node/chokaku_node.hpp"
#include <csignal>
#include <string>

namespace {
    rclcpp::Node::SharedPtr g_node = nullptr;
    
    void signal_handler(int sig) {
        std::cout << "\nStopping OpenVINO classification...\n";
        if (g_node) {
            rclcpp::shutdown();
        }
    }
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    rclcpp::init(argc, argv);
    
    g_node = std::make_shared<ChokakuNode>();
    
    try {
        rclcpp::spin(g_node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(g_node->get_logger(), "Error in node execution: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}