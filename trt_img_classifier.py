import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
from PIL import Image
import time
from torchvision import transforms
from torchvision.models import ResNet50_Weights

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# CUDA başlatma
def initialize_cuda():
    cuda.init()
    device = cuda.Device(0)  # İlk GPU aygıtını seç
    context = device.make_context()
    return context

# CUDA bağlamını temizleme
def clean_cuda_context(cuda_context):
    if cuda_context:
        cuda_context.pop()

# TensorRT Engine yükleme fonksiyonu
def load_engine(engine_path):
    try:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"Error loading engine: {e}")
        return None

# TensorRT Engine ve Execution Context ayarları
engine = load_engine('/home/ertugrul/models_ws/src/comp_models/models/trt_model.trt')
if engine is None:
    raise SystemExit("TensorRT engine couldn't be loaded. Exiting.")



context = engine.create_execution_context()

# Sınıf etiketlerini yükle
weights = ResNet50_Weights.DEFAULT
class_names = weights.meta["categories"]

# Görüntü işleme için transform
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

# CUDA bellek yönetimi fonksiyonu
def allocate_buffers(engine):
    try:
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name= engine.get_tensor_name(i)
            print(f"Tensor {i}: Name: {tensor_name}, Shape: {engine.get_tensor_shape(tensor_name)}, Dtype: {engine.get_tensor_dtype(tensor_name)}")
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            buffer = cuda.mem_alloc(size * dtype().itemsize)
            bindings.append(int(buffer))
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(buffer)
            else:
                outputs.append(buffer)
        return inputs, outputs, bindings, stream
    except Exception as e:
        print(f"Error allocating buffers: {e}")
        return None, None, None, None

# TensorRT ile sınıflandırma fonksiyonu
def classify_image_tensorrt(image_path, cuda_context):
    try:
        # Görseli yükle ve ön işleme
        img = Image.open(image_path)
        img_t = transform(img).unsqueeze(0).numpy()
        print(f"Input shape: {img_t.shape}")


        # TensorRT için giriş/çıkış buffer'larını ayarla
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        if inputs is None or outputs is None:
            raise RuntimeError("Failed to allocate buffers.")

        # Girdiyi GPU belleğine kopyala
        cuda.memcpy_htod_async(inputs[0], img_t, stream)


        for i in range(engine.num_io_tensors):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])


        # Model ile tahmin yaparken süreyi ölç
        start_time = time.time()
        N = 1000  # 1000 kere döngü
        for _ in range(N):
            context.execute_async_v3(stream_handle=stream.handle)

        # Çıkışı CPU'ya kopyala
        output_shape = context.get_tensor_shape(1)
        output = np.zeros(trt.volume(output_shape), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, outputs[0], stream)
        stream.synchronize()  # Asenkron işlemi senkronize et
        end_time = time.time()

        # Ortalama tahmin süresi hesapla
        avg_time = (end_time - start_time) / N
        print(f"Avg Inference Time (TensorRT): {(avg_time * 1000):.6f} ms")

    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        clean_cuda_context(cuda_context)

# CUDA başlat ve görseli sınıflandır
cuda_context = initialize_cuda()
classify_image_tensorrt("/home/ertugrul/models_ws/src/comp_models/images/Cat_November_2010-1a.jpg", cuda_context)