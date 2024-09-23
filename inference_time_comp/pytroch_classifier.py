import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
weights = ResNet50_Weights.DEFAULT
model = torch.load('models/model_weights.pth').to(device)
model.eval()
print(f"Model is on device: {next(model.parameters()).device}")

# Görseli yükleme ve ön işleme (transformlar)
img = Image.open("images/Cat_November_2010-1a.jpg")
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0).to(device)

# Inference süresini ölçme
start_time = time.time()
N = 1000
for _ in range(N):
  with torch.no_grad():
    outputs = model(batch)
end_time = time.time()

avg_time = (end_time - start_time) / N


# Süreyi yazdırın

print(f"Inference süresi: {(1000*avg_time):.6f} ms")
