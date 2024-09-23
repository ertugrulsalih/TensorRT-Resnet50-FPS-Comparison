import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import time
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

# ONNX model ve GPU sağlayıcısını ayarla
providers = ['TensorrtExecutionProvider']  # GPU için CUDA kullanın
ort_session = ort.InferenceSession('/home/ertugrul/models_ws/src/comp_models/models/onnx_model.onnx', providers=providers)

# ResNet50 sınıf etiketlerini yükle
weights = ResNet50_Weights.DEFAULT
class_names = weights.meta["categories"]

# Görüntüleri uygun formata dönüştürmek için transform
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# FPS hesaplamaları için zaman başlangıcı
fps_start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS hesaplaması için frame sayısını arttır
    frame_count += 1

    # Görüntüyü dönüştür
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV görüntüsünü PIL formatına çevir
    img_t = transform(img).unsqueeze(0).numpy()  # Görüntüyü model için uygun formata getir ve numpy array'e çevir

    # Model ile tahmin yap
    ort_inputs = {ort_session.get_inputs()[0].name: img_t}
    ort_outs = ort_session.run(None, ort_inputs)

    # En yüksek olasılıklı sınıfı bul
    ort_outs = np.array(ort_outs[0])
    predicted_class = np.argmax(ort_outs, axis=1)
    predicted_label = class_names[predicted_class[0]]

    # FPS'i hesapla
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = frame_count / time_diff

    # Tahmin edilen sınıfı ve FPS'i ekranda göster
    cv2.putText(frame, f"Class: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Kamera', frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat ve pencereleri temizle
cap.release()
cv2.destroyAllWindows()
