import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
import matplotlib.patches as patches
import os
from ultralytics.nn.tasks import DetectionModel  # Import the custom class
from ultralytics import YOLO


def load_model(model_path, device):
    """Load the PyTorch model from the .pth file."""

    model = YOLO(model_path)  # Load the YOLO model
    model.to(device)
    return model

def preprocess_image(image_path):
    """Preprocess the image for the model."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def display_bboxes(image_path, bboxes):
    """Display bounding boxes on the image."""
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox in bboxes:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")
    models_dir = os.path.join(script_dir, "../models")
    
    
    parser = argparse.ArgumentParser(description="Run model on an image and display bounding boxes.")
    parser.add_argument("--model", "-m", required=True, help="Path to the .pth model file.")
    parser.add_argument("--image", "-i", required=True, help="Path to the image file.")
    args = parser.parse_args()
    
    

    # Load model and preprocess image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = os.path.join(models_dir, args.model)
    img_path = os.path.join(data_dir, args.image)
    
    model = load_model(model_path, device) 
    image_tensor = preprocess_image(img_path)
    

    # Run the model on the image
    results = model(image_tensor)
    
    # Assuming the model outputs bounding boxes in the format [x, y, width, height]
    bboxes = results[0].boxes.xywh.cpu().numpy()  # Get the bounding boxes from the model output

    # Display bounding boxes
    display_bboxes(img_path, bboxes)

if __name__ == "__main__":
    main()