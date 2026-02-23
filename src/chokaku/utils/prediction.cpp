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

#include "chokaku/utils/prediction.hpp"

namespace chokaku {

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

} // namespace chokaku
