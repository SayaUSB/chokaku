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
