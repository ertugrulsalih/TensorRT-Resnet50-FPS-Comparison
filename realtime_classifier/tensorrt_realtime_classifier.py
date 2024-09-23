import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
from PIL import Image
import pycuda.autoinit
import cv2
import time
from torchvision import transforms
from torchvision.models import ResNet50_Weights

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# TensorRT Engine yükleme fonksiyonu
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# TensorRT Engine ve Execution Context ayarları
engine = load_engine('models/trt_model.trt')
context = engine.create_execution_context()

# Sınıf etiketlerini yükle
weights = ResNet50_Weights.DEFAULT
class_names = weights.meta["categories"]

# Görüntü işleme için transform
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])


# CUDA bellek yönetimi fonksiyonu
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        buffer = cuda.mem_alloc(size * dtype().itemsize)
        bindings.append(int(buffer))
        if engine.binding_is_input(binding):
            inputs.append(buffer)
        else:
            outputs.append(buffer)
    return inputs, outputs, bindings, stream

# TensorRT ile sınıflandırma fonksiyonu
def classify_frame_tensorrt(frame):
    # Görüntüyü ön işleme tabi tut
    img = Image.fromarray(frame)
    img_t = transform(img).unsqueeze(0).numpy().astype(np.float32)

    # TensorRT için giriş/çıkış buffer'larını ayarla
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Girdiyi GPU belleğine kopyala
    cuda.memcpy_htod_async(inputs[0], img_t, stream)

    # Model ile tahmin yap
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Çıkışı CPU'ya kopyala
    output_shape = engine.get_binding_shape(1)
    output = np.empty(trt.volume(output_shape), dtype=np.float32)
    cuda.memcpy_dtoh_async(output, outputs[0], stream)
    stream.synchronize()

    # Sınıflandırma sonucu ve sınıf adını döndür
    class_idx = np.argmax(output)
    class_name = class_names[class_idx]
    return class_name


# Kameradan veri al ve gerçek zamanlı sınıflandırma yap
def real_time_classifier():
    cap = cv2.VideoCapture(0)  # Kamerayı aç

    if not cap.isOpened():
        print("Kamera açılamadı")
        return

    fps_start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()  # Kameradan görüntü al
        if not ret:
            break

        frame_count += 1

        class_name = classify_frame_tensorrt(frame)  # Görüntüyü sınıflandır

        # FPS hesapla
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = frame_count / time_diff

        # Sınıf adını ve FPS'yi görüntüle
        cv2.putText(frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Görüntüyü ekranda göster
        cv2.imshow('Real-Time Classification', frame)

        # 'q' tuşuna basarak çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Gerçek zamanlı sınıflandırmayı başlat
real_time_classifier()