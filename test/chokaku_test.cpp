#include "chokaku/chokaku.hpp"
#include <iostream>
#include <portaudio.h>

int main() {
    // Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio initialization failed: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }
    
    std::cout << "Available Audio Devices:\n";
    std::cout << "========================\n\n";
    
    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "Error getting device count: " << Pa_GetErrorText(numDevices) << "\n";
        Pa_Terminate();
        return 1;
    }
    
    if (numDevices == 0) {
        std::cout << "No audio devices found.\n";
        Pa_Terminate();
        return 0;
    }
    
    const PaDeviceInfo* deviceInfo;
    
    for (int i = 0; i < numDevices; i++) {
        deviceInfo = Pa_GetDeviceInfo(i);
        
        std::cout << "Device ID: " << i << "\n";
        std::cout << "Name: " << deviceInfo->name << "\n";
        std::cout << "Host API: " << Pa_GetHostApiInfo(deviceInfo->hostApi)->name << "\n";
        std::cout << "Max Input Channels: " << deviceInfo->maxInputChannels << "\n";
        std::cout << "Max Output Channels: " << deviceInfo->maxOutputChannels << "\n";
        std::cout << "Default Sample Rate: " << deviceInfo->defaultSampleRate << "\n";
        
        if (deviceInfo->maxInputChannels > 0) {
            std::cout << "Default Low Input Latency: " << deviceInfo->defaultLowInputLatency << " seconds\n";
            std::cout << "Default High Input Latency: " << deviceInfo->defaultHighInputLatency << " seconds\n";
        }
        
        if (deviceInfo->maxOutputChannels > 0) {
            std::cout << "Default Low Output Latency: " << deviceInfo->defaultLowOutputLatency << " seconds\n";
            std::cout << "Default High Output Latency: " << deviceInfo->defaultHighOutputLatency << " seconds\n";
        }
        
        // Mark default devices
        if (i == Pa_GetDefaultInputDevice()) {
            std::cout << "*** DEFAULT INPUT DEVICE ***\n";
        }
        if (i == Pa_GetDefaultOutputDevice()) {
            std::cout << "*** DEFAULT OUTPUT DEVICE ***\n";
        }
        
        std::cout << "------------------------\n\n";
    }
    
    std::cout << "Default Input Device ID: " << Pa_GetDefaultInputDevice() << "\n";
    std::cout << "Default Output Device ID: " << Pa_GetDefaultOutputDevice() << "\n";
    
    // Show recommended input devices
    std::cout << "\nRecommended Input Devices (with input channels):\n";
    std::cout << "===============================================\n";
    for (int i = 0; i < numDevices; i++) {
        deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            std::cout << "ID " << i << ": " << deviceInfo->name 
                      << " (" << deviceInfo->maxInputChannels << " channels)\n";
        }
    }
    
    Pa_Terminate();
    return 0;
}