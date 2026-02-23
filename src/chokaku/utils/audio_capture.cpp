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

#include "chokaku/utils/audio_capture.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace chokaku {

AudioCapture::AudioCapture(const ChokakuConfig& config)
    : config_(config) {
    
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw std::runtime_error(std::string("PortAudio initialization failed: ") + Pa_GetErrorText(err));
    }
}

AudioCapture::~AudioCapture() {
    Pa_Terminate();
}

std::vector<float> AudioCapture::capture_audio_blocking(float duration, int sample_rate) {
    size_t num_samples = static_cast<size_t>(duration * sample_rate);
    std::vector<float> buffer(num_samples);
    
    PaStream* stream;
    
    // Set up input parameters
    PaStreamParameters inputParams;
    
    // Validate the specified device
    if (config_.mic_id >= 0 && config_.mic_id < Pa_GetDeviceCount()) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(config_.mic_id);
        if (deviceInfo->maxInputChannels > 0) {
            inputParams.device = config_.mic_id;
        } else {
            std::cerr << "Device " << config_.mic_id << " (" << deviceInfo->name 
                      << ") has no input channels, falling back to default input device\n";
            inputParams.device = Pa_GetDefaultInputDevice();
        }
    } else {
        std::cerr << "Invalid device ID " << config_.mic_id 
                  << ", falling back to default input device\n";
        inputParams.device = Pa_GetDefaultInputDevice();
    }
    
    // Check if we have a valid device
    if (inputParams.device == paNoDevice) {
        std::cerr << "No valid input device found\n";
        return {};
    }
    
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(inputParams.device);
    std::cout << "Using device: " << deviceInfo->name << " (max input channels: " 
              << deviceInfo->maxInputChannels << ")\n";
    
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = deviceInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;
    
    PaError err = Pa_OpenStream(&stream,
                              &inputParams,     // input parameters
                              nullptr,          // output parameters (none)
                              sample_rate,
                              256,             // frames per buffer
                              paClipOff,       // stream flags
                              nullptr,         // callback (blocking mode)
                              nullptr);
    
    if (err != paNoError) {
        std::cerr << "Failed to open PortAudio stream: " << Pa_GetErrorText(err) << "\n";
        return {};
    }
    
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Failed to start stream: " << Pa_GetErrorText(err) << "\n";
        Pa_CloseStream(stream);
        return {};
    }
    
    // Read audio
    err = Pa_ReadStream(stream, buffer.data(), num_samples);
    if (err != paNoError && err != paInputOverflowed) {
        std::cerr << "Error reading audio: " << Pa_GetErrorText(err) << "\n";
    }
    
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    
    return buffer;
}

int AudioCapture::pa_callback(const void* input_buffer, void* output_buffer,
                              unsigned long frames_per_buffer,
                              const PaStreamCallbackTimeInfo* time_info,
                              PaStreamCallbackFlags status_flags,
                              void* user_data) {
    // Callback implementation for non-blocking mode if needed
    return paContinue;
}

} // namespace chokaku
