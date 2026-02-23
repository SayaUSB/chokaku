# Chokaku - ROS2 Whistle Detection Node

A ROS2 package for real-time whistle detection using OpenVINO inference engine.

## Overview

Chokaku is a high-performance whistle detection system that:
- Captures audio in real-time using PortAudio
- Processes audio through OpenVINO inference models
- Publishes whistle detection events via ROS2 topics
- Provides configurable confidence thresholds and output options

## Features

- **Real-time Audio Processing**: Continuous audio capture and classification
- **OpenVINO Acceleration**: Hardware-accelerated inference for optimal performance
- **Configurable Detection**: Adjustable confidence threshold for whistle detection
- **ROS2 Integration**: Standard ROS2 topics and node lifecycle
- **Flexible Output**: Configurable number of top predictions and console output
- **Clean Shutdown**: Proper signal handling and thread management

## Installation

### Prerequisites

- ROS2 Jazzy
- OpenVINO Runtime
- PortAudio
- C++17 compatible compiler
- CMake 3.16+

### Building from Source

```bash
# Build the package
colcon build --packages-select chokaku

# Source the workspace
source install/setup.zsh
```

## Usage

### Running the Main Node

```bash
# Start whistle detection node
ros2 run chokaku main

# With custom config
ros2 run chokaku main --ros-args -p config_path:=/path/to/config.json
```

### Running Tests

```bash
# Run test suite
ros2 run chokaku check
```

## Configuration

The node uses a JSON configuration file (default: `config/chokaku_config.json`):

```json
{
  "mic_id": 4,
  "model_dir": "./models",
  "model_xml": "chokaku.xml",
  "model_bin": "chokaku.bin",
  "model_class_map": "chokaku_class_map.csv",
  "sample_rate": 16000,
  "chunk_duration": 0.96,
  "audio_buffer_size": 512,
  "confidence_threshold": 0.3,
  "top_predictions_output": 5,
  "print_output": true
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|--------|----------|-------------|
| `mic_id` | int | 4 | Audio device ID for microphone input |
| `model_dir` | string | "./models" | Directory containing model files |
| `model_xml` | string | "chokaku.xml" | OpenVINO model XML file |
| `model_bin` | string | "chokaku.bin" | OpenVINO model weights file |
| `model_class_map` | string | "chokaku_class_map.csv" | Class name mapping CSV file |
| `sample_rate` | int | 16000 | Audio sampling rate in Hz |
| `chunk_duration` | float | 0.96 | Audio chunk duration in seconds |
| `audio_buffer_size` | int | 512 | Audio buffer size |
| `confidence_threshold` | float | 0.3 | Minimum confidence for whistle detection |
| `top_predictions_output` | int | 5 | Number of top predictions to display |
| `print_output` | bool | true | Enable/disable console output |

## ROS2 Topics

### Published Topics

- `/chokaku/detection` (`std_msgs/Bool`)
  - Published when whistle detection state changes
  - `true`: Whistle detected
  - `false`: Whistle detection ended

### Parameters

- `config_path` (string): Path to configuration file

## Model Information

The system uses pre-trained audio classification models to detect:
- **Whistling** (class index 35)
- **Whistle** (class index 396)

Models are trained on a large dataset of audio events and provide confidence scores for all detected classes.

## Output Examples

### Console Output (when `print_output: true`)

```
1. Speech: 0.8542
2. Inside_Small_Room: 0.0421
3. Narration: 0.0187
4. Television: 0.0123
5. Conversation: 0.0089
```

### ROS2 Log Output

```
[INFO] [chokaku_node]: Whistle detected! Class: Whistling (index: 35, confidence: 0.841, threshold: 0.300)
[INFO] [chokaku_node]: Whistle detection ended
```

## Troubleshooting

### Common Issues

1. **Audio Device Not Found**
   - Check `mic_id` in config matches available devices
   - Use `arecord -l` to list audio devices

2. **Model Files Not Found**
   - Verify model files exist in `model_dir`
   - Check file permissions

3. **Segmentation Fault on Shutdown**
   - Ensure clean shutdown with Ctrl+C
   - Check audio device compatibility

4. **Low Detection Accuracy**
   - Adjust `confidence_threshold`
   - Check microphone placement and background noise

### Debug Mode

Set `"print_output": false` to suppress console output and only see ROS2 logs:

```bash
# Minimal output for production use
ros2 run chokaku main --ros-args -p print_output:=false
```

## Performance

- **Real-time Processing**: ~1 second latency
- **CPU Usage**: Low (OpenVINO optimized)
- **Memory Usage**: ~200MB (model dependent)
- **Audio Quality**: 16kHz, mono input

## License

MIT License
