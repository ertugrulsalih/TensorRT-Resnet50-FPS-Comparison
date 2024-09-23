import torch
from torch2trt import torch2trt
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelin yüklenmesi
weights = ResNet50_Weights.DEFAULT
model = torch.load('models/model_weights.pth').to(device)
model.eval()

# Modeli Torch2TRT ile dönüştürme
model_trt = torch2trt(model, [torch.randn(1, 3, 224, 224).to(device)])

# Görseli yükleme ve ön işleme (transformlar)
img = Image.open("images/Cat_November_2010-1a.jpg")
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0).to(device)

# 1. Torch CUDA ile Inference süresini ölçme
start_time = time.time()
N = 1000
for _ in range(N):
    with torch.no_grad():
        outputs = model(batch)
end_time = time.time()

avg_time_torch = (end_time - start_time) / N
print(f"Torch CUDA Inference süresi: {(1000*avg_time_torch):.6f} ms")

# 2. Torch2TRT ile Inference süresini ölçme
start_time = time.time()
for _ in range(N):
    with torch.no_grad():
        outputs_trt = model_trt(batch)
end_time = time.time()

avg_time_trt = (end_time - start_time) / N
print(f"Torch2TRT Inference süresi: {(1000*avg_time_trt):.6f} ms")

# Kıyaslama
speedup = avg_time_torch / avg_time_trt
print(f"Inference Hız Kazancı: {speedup:.2f}x")

