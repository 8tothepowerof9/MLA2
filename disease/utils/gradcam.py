import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from src.classify_diseases.models.resnet_model import ResNet18Classifier
from src.classify_diseases.models.cbam import ResNet34CBAMClassifier

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient)
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.cpu().numpy()

        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()



def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig = image.copy()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, orig

def overlay_heatmap(cam, image, alpha=0.4):
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return overlay



# Load model
model = ResNet34CBAMClassifier(num_classes=10)  # set correct num_classes
# model = ResNet18Classifier(num_classes=10)  # set correct num_classes
# model.load_state_dict(torch.load("checkpoints/classify_diseases/model_masked_cbam34.pt", map_location="cpu"))
model.load_state_dict(torch.load("checkpoints/model_cbam.pt", map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Pick target layer (usually last conv layer)
target_layer = model.base_model.layer4[-1] 

# Preprocess
image_path = "data/test_images/200020.jpg"
input_tensor, original_image = preprocess_image(image_path)
input_tensor = input_tensor.to(device)

# Grad-CAM
cam_extractor = GradCAM(model, target_layer)
cam = cam_extractor.generate(input_tensor)

# Overlay
overlay = overlay_heatmap(cam, original_image)

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.tight_layout()
plt.show()
cv2.imwrite("gradcam_output.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
