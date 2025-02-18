from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# List of class names for diseases
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Best thresholds for prediction
best_thresholds = [
    0.66, 0.71, 0.66, 0.63, 0.70, 0.69, 0.69, 0.69, 0.67, 0.70, 0.71, 0.72, 0.69, 0.71
]

# Define the model
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

        # Register hooks to save activations and gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, class_idx):
        if self.gradients is None or self.activation is None:
            raise ValueError("Gradients and activations are not available.")
        
        weights = self.gradients.mean(dim=[0, 2, 3]).to(self.activation.device)
        activation = self.activation.squeeze(0).detach()
        heatmap = torch.zeros(activation.shape[1:], device=activation.device)

        for i, w in enumerate(weights):
            heatmap += w * activation[i]

        heatmap = torch.clamp(heatmap, min=0)
        return heatmap.cpu().numpy()

# Function to process image
def process_image(image_input):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("Unsupported image input type. Provide a file path or PIL Image.")

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return input_tensor, image

# Load model globally for reuse
N_CLASSES = len(CLASS_NAMES)

# Initialize FastAPI app
app = FastAPI()

# Model loading and Grad-CAM setup at startup
model = None
grad_cam = None

@app.on_event("startup")
async def startup_event():
    global model, grad_cam
    # Load the model once when the application starts
    model = DenseNet121(N_CLASSES)
    checkpoint = torch.load("model.pth.tar", map_location=torch.device('cpu'))
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "").replace(".1", "1").replace(".2", "2").replace(".3", "3")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Set up Grad-CAM
    target_layer = model.densenet121.features[-1]  # Last convolutional layer of DenseNet121
    grad_cam = GradCAM(model, target_layer)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    
    # Process the image and predict diseases
    input_tensor, original_image = process_image(image)
    probabilities = predict_diseases(model, input_tensor)
    
    # Display predictions
    detected_diseases = [
        f"{CLASS_NAMES[i]} ({probabilities[i]:.2%})"
        for i in range(len(CLASS_NAMES))
        if probabilities[i] > best_thresholds[i]
    ]
    detected_text = "\n".join(detected_diseases) if detected_diseases else "No diseases detected."
    print(f"Predictions:\n{detected_text}")
    
    # Generate Grad-CAM visualization
    grad_cam_output = grad_cam_on_predictions(model, input_tensor, original_image, CLASS_NAMES, best_thresholds)

    return StreamingResponse(grad_cam_output, media_type="image/png")

# Run the FastAPI app locally (optional for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
