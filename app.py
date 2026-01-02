import streamlit as st
import cv2
import torch
import numpy as np
import dlib # Make sure dlib is installed
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn as nn
import torchvision.models as models

# 1. Model Definition (Must be identical to the training script)
class Network(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18() # No pretrained=True here, as we load our own weights
        # Modify the first convolutional layer for single-channel grayscale input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final fully connected layer for 136 landmark outputs
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# 2. Load Model and Dlib Detector
@st.cache_resource # Cache the model loading for better performance
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)
    model.load_state_dict(torch.load('face_landmarks.pth', map_location=device))
    model.eval()
    detector = dlib.get_frontal_face_detector()
    return model, device, detector

model, device, detector = load_model()

# 3. Inference Function
def predict_landmarks(image_pil, model, detector, device):
    # Convert PIL image to OpenCV format (grayscale)
    image_np = np.array(image_pil.convert('L'))

    # Use dlib to detect faces
    detects = detector(image_np, 1)

    if len(detects) == 0:
        st.warning("No face detected. Please try another image.")
        return None, None

    # Take the first detected face
    intersect = detects[0]
    x1, y1, x2, y2 = intersect.left(), intersect.top(), intersect.right(), intersect.bottom()

    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_np.shape[1], x2)
    y2 = min(image_np.shape[0], y2)

    # Crop the face from the original PIL image
    cropped_face_pil = image_pil.crop((x1, y1, x2, y2))

    # Resize to model input size (224x224)
    input_image_pil = cropped_face_pil.resize((224, 224))

    # Preprocess the image for the model
    image_tensor = TF.to_tensor(input_image_pil.convert('L')) # Convert to grayscale tensor
    image_tensor = TF.normalize(image_tensor, [0.5], [0.5]) # Normalize to [-0.5, 0.5]
    image_tensor = image_tensor.unsqueeze(0).to(device) # Add batch dimension

    # Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    # Denormalize predictions
    # Landmarks were normalized to [-0.5, 0.5] relative to 224x224
    predictions = (predictions.cpu().view(-1, 2) + 0.5) * 224

    # Adjust landmarks to original cropped face size before final image display
    # We need to scale landmarks from 224x224 back to the cropped_face_pil dimensions
    scale_x = cropped_face_pil.width / 224.0
    scale_y = cropped_face_pil.height / 224.0
    predictions[:, 0] = predictions[:, 0] * scale_x
    predictions[:, 1] = predictions[:, 1] * scale_y

    return cropped_face_pil, predictions.numpy() # Return PIL image and numpy landmarks

# Streamlit UI
st.title("Facial Landmark Detection")
st.write("Upload an image or use your webcam to detect facial landmarks.")

# Input choice
input_choice = st.radio("Choose input method:", ("Upload Image", "Webcam"))

image_file = None
if input_choice == "Upload Image":
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
elif input_choice == "Webcam":
    st.warning("Webcam support in Streamlit Cloud is experimental. For local deployment, ensure you have permissions.")
    st.markdown("Due to Colab environment limitations, webcam functionality might not work directly within Streamlit hosted here. For local deployment, use `streamlit run app.py` and access from your browser.")
    # This part would typically be more complex for a robust webcam integration in pure Streamlit
    # For simplicity, we will disable active webcam capture in Colab
    st.write("Please upload an image instead for Colab environment.")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="webcam_fallback")

if image_file is not None:
    # Read the image
    image_pil = Image.open(image_file)
    st.image(image_pil, caption='Original Image', use_column_width=True)

    st.subheader("Processing...")

    # Perform prediction
    cropped_face_pil, landmarks = predict_landmarks(image_pil, model, detector, device)

    if cropped_face_pil is not None and landmarks is not None:
        # Convert cropped PIL image to numpy for drawing
        cropped_face_np = np.array(cropped_face_pil.convert('RGB'))

        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(cropped_face_np, (int(x), int(y)), 2, (0, 255, 0), -1) # Green dots

        st.image(cropped_face_np, caption='Detected Face with Landmarks', use_column_width=True)
