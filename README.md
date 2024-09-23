# Real-Time Classifier With TensorRT

This work was conducted on a laptop equipped with Ubuntu 20.04 and an Nvidia GTX 1070 graphics card. The outcomes of the live image classification test are presented below. The study encompasses a performance comparison utilising a pre-existing PyTorch model, converting the .pth format model to ONNX format, and evaluating both models with ONNX Runtime. The results are expressed in frames per second (FPS) values on both the central processing unit (CPU) and the graphics processing unit (GPU) of the aforementioned laptop.


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
