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
    stop_stream();
    Pa_Terminate();
}

void AudioCapture::start_stream(int sample_rate) {
    if (stream_) return;

    PaStreamParameters inputParams;
    if (config_.mic_id >= 0 && config_.mic_id < Pa_GetDeviceCount()) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(config_.mic_id);
        if (deviceInfo->maxInputChannels > 0) {
            inputParams.device = config_.mic_id;
        } else {
            inputParams.device = Pa_GetDefaultInputDevice();
        }
    } else {
        inputParams.device = Pa_GetDefaultInputDevice();
    }

    if (inputParams.device == paNoDevice) {
        throw std::runtime_error("No valid input device found");
    }

    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(inputParams.device);
    std::cout << "Opening persistent stream on device: " << deviceInfo->name << "\n";

    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = deviceInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(&stream_, &inputParams, nullptr, sample_rate, 256, paClipOff, nullptr, nullptr);
    if (err != paNoError) {
        throw std::runtime_error(std::string("Failed to open PortAudio stream: ") + Pa_GetErrorText(err));
    }

    err = Pa_StartStream(stream_);
    if (err != paNoError) {
        Pa_CloseStream(stream_);
        stream_ = nullptr;
        throw std::runtime_error(std::string("Failed to start stream: ") + Pa_GetErrorText(err));
    }
}

void AudioCapture::stop_stream() {
    if (stream_) {
        Pa_StopStream(stream_);
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }
}

std::vector<float> AudioCapture::capture_audio_blocking(float duration, int sample_rate) {
    if (!stream_) {
        start_stream(sample_rate);
    }

    size_t num_samples = static_cast<size_t>(duration * sample_rate);
    std::vector<float> buffer(num_samples);
    
    PaError err = Pa_ReadStream(stream_, buffer.data(), num_samples);
    if (err != paNoError && err != paInputOverflowed) {
        std::cerr << "Error reading audio: " << Pa_GetErrorText(err) << "\n";
        // Attempt to recover if it's a fatal error
        if (err == paStreamIsStopped || err == paNotInitialized) {
            stop_stream();
        }
    }
    
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
