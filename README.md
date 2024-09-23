# Real-Time Classifier With TensorRT

This work was conducted on a laptop equipped with Ubuntu 20.04 and an Nvidia GTX 1070 graphics card. The outcomes of the real-time image classification test are presented below. The study involves a performance comparison by utilizing a pre-trained PyTorch model, converting the model from .pth to ONNX format, and evaluating both models using ONNX Runtime and TensorRT. The results are measured in frames per second (FPS) on both the CPU and GPU of the system. This study highlights the efficiency of deep learning models in real-time object detection and their performance across different frameworks and inference engines.


## Project Structure

- **onnx2trt_realtime_class.py**: Real-time classification via ONNX Runtime with TensorRT Provider.
- **tensorrt_realtime_class.py**: Real-time classification directly with TensorRT.
- **torch2trt_realtime_class.py**: Real-time classification via PyTorch with Torch2TRT library.

## System Requirements

- **Ubuntu**: 20.04
- **CUDA**: 11.8
- **cuDNN**: 8.9.x
- **TensorRT**: 8.6.1
- **PyTorch**: 2.0.0
- **Torchvision**: 0.18
- **Torch2TRT**: 0.5
- **ONNX**: 1.16.2
- **ONNX Runtime-GPU**: 1.17.0
- **PyCUDA**: 2024.1.2

## Performance Results

| Device     | PyTorch RT Classifier (model_weights.pth) | ONNX Runtime RT Classifier (onnx_model.onnx) |
|-----------|--------------------------------------------------|----------------------------------------------------|
| CPU       | ~5 FPS                                           | ~18 FPS                                            |
| CUDA      | ~29 FPS                                          | ~27 FPS                                            |
| TensorRT  | ~30 FPS (torch2trt)                              | ~29 FPS (TensorRT Provider)                        |
