import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
IMAGE_SIZE = 224
MODEL_PATH = "C:/Users/sidha/OneDrive/Desktop/Software-Dev/plant_disease_model.pth"
gradients = None
activations = None
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image
def save_gradient_hook(grad):
    global gradients
    gradients = grad
def forward_hook(module, input, output):
    global activations
    activations = output
    output.register_hook(save_gradient_hook)
def predict_image_with_gradcam(image_path):
    global gradients, activations
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    num_classes = state_dict['fc.weight'].shape[0]
    classes = [f"Class_{i}" for i in range(num_classes)]  
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    model.layer4.register_forward_hook(forward_hook)
    input_tensor, original_image = preprocess_image(image_path)
    output = model(input_tensor)
    pred_class = output.argmax().item()
    pred_label = classes[pred_class]
    model.zero_grad()
    output[0, pred_class].backward()
    pooled_gradients = torch.mean(gradients[0], dim=[1, 2])
    activation_map = activations[0]
    for i in range(activation_map.shape[0]):
        activation_map[i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activation_map, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
    gradcam_path = os.path.join("static", "uploads", "gradcam_result.jpg")
    os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
    cv2.imwrite(gradcam_path, superimposed_img)
    return pred_label, gradcam_path
