#ifndef CHOKAKU_AUDIO_CAPTURE_HPP
#define CHOKAKU_AUDIO_CAPTURE_HPP

#include <portaudio.h>
#include <vector>
#include "chokaku/utils/config.hpp"

namespace chokaku {

class AudioCapture {
public:
    AudioCapture(const ChokakuConfig& config);
    ~AudioCapture();
    
    std::vector<float> capture_audio_blocking(float duration, int sample_rate);
    
private:
    ChokakuConfig config_;
    
    static int pa_callback(const void* input_buffer, void* output_buffer,
                          unsigned long frames_per_buffer,
                          const PaStreamCallbackTimeInfo* time_info,
                          PaStreamCallbackFlags status_flags,
                          void* user_data);
};

} // namespace chokaku

#endif // CHOKAKU_AUDIO_CAPTURE_HPP
