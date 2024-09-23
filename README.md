# Gerçek Zamanlı Sınıflandırıcı Projesi

Bu proje, TensorRT, ONNX ve Torch2TRT kullanarak gerçek zamanlı nesne tespiti yapmayı amaçlamaktadır. Üç farklı yöntemi kullanarak görüntü sınıflandırma performansını karşılaştırır.

## Proje Yapısı

- **onnx2trt_realtime_class.py**: ONNX Runtime kullanarak kamera ile gerçek zamanlı nesne tespiti.
- **tensorrt_realtime_class.py**: Doğrudan TensorRT ile gerçek zamanlı sınıflandırma.
- **torch2trt_realtime_class.py**: Torch2TRT kütüphanesi ile TensorRT üzerinden gerçek zamanlı sınıflandırma.

## Sistem Gereksinimleri

- **CUDA**: 11.8
- **cuDNN**: 8.9.x
- **TensorRT**: 8.6.1
- **PyTorch**: 2.0.0
- **Torchvision**: 0.18
- **Torch2TRT**: 0.5
- **ONNX**: 1.16.2
- **ONNX Runtime-GPU**: 1.17.0
- **PyCUDA**: 2024.1.2
