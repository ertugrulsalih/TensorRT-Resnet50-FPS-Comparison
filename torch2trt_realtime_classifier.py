import torch
from torch2trt import torch2trt
from torchvision.models import ResNet50_Weights
import cv2
import time
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
weights = ResNet50_Weights.DEFAULT
model = torch.load('comp_models/models/model_weights.pth').to(device)
model.eval()

# Convert the model to TensorRT using torch2trt
model_trt = torch2trt(model, [torch.randn(1, 3, 224, 224).to(device)])

# Check the device where the model is loaded
print(f"Model is on device: {next(model.parameters()).device}")

# Start the camera
cap = cv2.VideoCapture(0)

# Initialize time for FPS calculation
fps_start_time = time.time()
frame_count = 0

# Transform for input images
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame for model input
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    preprocess = weights.transforms()

    # Preprocess the input and move it to the GPU
    batch = preprocess(img).unsqueeze(0).to(device)
    frame_count += 1

    # Run inference using TensorRT model
    with torch.no_grad():
        outputs = model_trt(batch)
    
    # Get the predicted class
    prediction = model_trt(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    # Calculate FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = frame_count / time_diff

    # Display class name and FPS on screen
    cv2.putText(frame, f"Class: {category_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Camera', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
