import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import time
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

# Set up the ONNX model and GPU provider
providers = ['TensorrtExecutionProvider']  # Use TensorRT for GPU
ort_session = ort.InferenceSession('models/onnx_model.onnx', providers=providers)

# Load ResNet50 class labels
weights = ResNet50_Weights.DEFAULT
class_names = weights.meta["categories"]

# Define image transformations to preprocess the input
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

# Start the camera
cap = cv2.VideoCapture(0)

# Initialize time for FPS calculation
fps_start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Increase frame count for FPS calculation
    frame_count += 1

    # Convert the frame to the format needed for the model
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL format
    img_t = transform(img).unsqueeze(0).numpy()  # Apply transformations and convert to numpy array

    # Run inference with the model
    ort_inputs = {ort_session.get_inputs()[0].name: img_t}
    ort_outs = ort_session.run(None, ort_inputs)

    # Find the class with the highest probability
    ort_outs = np.array(ort_outs[0])
    predicted_class = np.argmax(ort_outs, axis=1)
    predicted_label = class_names[predicted_class[0]]

    # Calculate FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = frame_count / time_diff

    # Display the predicted class and FPS on the screen
    cv2.putText(frame, f"Class: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Camera', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
