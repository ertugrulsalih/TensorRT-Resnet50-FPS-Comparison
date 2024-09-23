import onnxruntime as ort
import numpy as np
from PIL import Image
import time
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

providers = ['CUDAExecutionProvider']
ort_session = ort.InferenceSession('/home/ertugrul/models_ws/src/comp_models/models/onnx_model.onnx', providers=providers)

weights = ResNet50_Weights.DEFAULT
class_names = weights.meta["categories"]

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])


def classify_image_onnx_gpu(image_path):

    img = Image.open(image_path)
    img_t = transform(img).unsqueeze(0).numpy()

    start_time = time.time()
    N = 1000
    for _ in range(N):
      ort_inputs = {ort_session.get_inputs()[0].name: img_t}
      ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.time()

    avg_time = (end_time - start_time) / N
    print(f"Avg Inference Time (CUDA Provider): {(avg_time*1000):.6f} ms")

classify_image_onnx_gpu("images/Cat_November_2010-1a.jpg")
