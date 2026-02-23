#ifndef CHOKAKU_AUDIO_PROCESSING_HPP
#define CHOKAKU_AUDIO_PROCESSING_HPP

#include <vector>

namespace chokaku {

std::vector<float> resample_audio(const std::vector<float>& waveform, 
                                 int src_rate, int dst_rate);
std::vector<float> preprocess_audio(const std::vector<float>& waveform, 
                                   int sample_rate = 16000,
                                   int target_sample_rate = 16000,
                                   float target_duration = 0.96f);

} // namespace chokaku

#endif // CHOKAKU_AUDIO_PROCESSING_HPP
