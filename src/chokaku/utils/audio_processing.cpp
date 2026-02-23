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

#include "chokaku/utils/audio_processing.hpp"
#include <algorithm>
#include <cmath>

namespace chokaku {

std::vector<float> resample_audio(const std::vector<float>& waveform, 
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

std::vector<float> preprocess_audio(const std::vector<float>& waveform, 
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

} // namespace chokaku
