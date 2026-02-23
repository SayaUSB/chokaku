#ifndef CHOKAKU_PREDICTION_HPP
#define CHOKAKU_PREDICTION_HPP

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include <fstream>
#include <sstream>

namespace chokaku {

struct PredictionResult {
    std::vector<std::string> predicted_classes;
    std::vector<float> confidence_scores;
    int top_class_index;
};

void print_prediction(const PredictionResult& result, size_t top_k = 3);
std::map<std::string, std::string> load_csv_class_map(const std::string& csv_path);

} // namespace chokaku

#endif // CHOKAKU_PREDICTION_HPP
