import torch
from torch2trt import torch2trt
from torchvision.models import ResNet50_Weights
import cv2
import time
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelin yüklenmesi
weights = ResNet50_Weights.DEFAULT
model = torch.load('comp_models/models/model_weights.pth').to(device)
model.eval()

# Modeli Torch2TRT ile dönüştürme
model_trt = torch2trt(model, [torch.randn(1, 3, 224, 224).to(device)])


# Modeli GPU'ya taşırken kullandığınız device'ı kontrol edin
print(f"Model is on device: {next(model.parameters()).device}")

# torch.save(model, 'model_weights.pth')

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# FPS hesaplamaları için zaman başlangıcı
fps_start_time = time.time()
frame_count = 0

# Görüntüleri uygun formata dönüştürmek için transform
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü dönüştür
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    preprocess = weights.transforms()

    # Giriş görüntüsünü dönüştür
    batch = preprocess(img).unsqueeze(0).to(device)  # Giriş tensorunu GPU'ya taşıyın
    frame_count += 1

    # Model ile tahmin yap
    with torch.no_grad():
        outputs = model_trt(batch)
    
    # Tahmin edilen sınıfı yazdır

    # Tahmin yap ve sonucu yazdır
    prediction = model_trt(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = frame_count / time_diff

    # Görüntüyü ekranda göster
    cv2.putText(frame, f"Class: {category_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Kamera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
