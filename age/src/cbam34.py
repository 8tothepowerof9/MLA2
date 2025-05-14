import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from models.cbam import ResNetAgeWithCBAM

# Constants for age normalization
AGE_MIN = 45.0
AGE_MAX = 82.0

# Transform (must match the one used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = ResNetAgeWithCBAM()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(image).item()
        age = output * (AGE_MAX - AGE_MIN) + AGE_MIN
    return round(age, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = load_model(args.model)
    age = predict(model, args.image)
    print(f"Predicted Age: {age} years")

if __name__ == "__main__":
    main()
