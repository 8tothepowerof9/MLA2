import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import cv2
from dataloader import RiceDataset
from models import *
from sklearn.metrics import f1_score
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import random
from PIL import Image


# Plot confusion matrix
def evaluate_and_plot_confusion_matrix(model, dataloader, class_names, device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = activations.sum(dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()


def show_gradcam_on_image(
    img_tensor, heatmap, predicted_class, actual_class, class_names, alpha=0.4
):
    # Convert to numpy image
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    # Resize heatmap and apply color
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) / 255.0

    overlay = heatmap_color * alpha + img
    overlay = np.clip(overlay, 0, 1)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(
        f"Grad-CAM\nPredicted: {class_names[predicted_class]} | Actual: {class_names[actual_class]}"
    )

    # Add colorbar with proper normalization
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap="jet")
    sm.set_array([])  # required for matplotlib >= 3.1
    fig.colorbar(sm, ax=ax, shrink=0.8, label="Attention Weight")

    plt.tight_layout()
    plt.show()


def show_random_prediction(
    dataset, model, class_names, device="cpu", filter_classes=None
):
    """
    Randomly selects an image from the dataset, performs inference, and plots it
    with predicted and actual labels.

    Args:
        dataset: PyTorch Dataset object
        model: Trained PyTorch model
        class_names: List of class name strings
        device: Device to run inference on ("cpu" or "cuda")
    """
    model.eval()

    # Get filtered indices
    if filter_classes:
        allowed_class_ids = [dataset.class_to_idx[c] for c in filter_classes]
        indices = [i for i, (_, y) in enumerate(dataset) if y in allowed_class_ids]
        if not indices:
            print(f"No images found for classes: {filter_classes}")
            return
    else:
        indices = list(range(len(dataset)))

    # Pick random index from filtered list
    idx = random.choice(indices)
    image, label = dataset[idx]
    image_tensor = image.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Prepare image for display
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(image_np)
    plt.axis("off")
    plt.title(
        f"Predicted: {class_names[predicted_class]}\nActual: {class_names[label]}"
    )
    plt.tight_layout()
    plt.show()


def print_macro_f1(val_dataloader, model):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to("cuda")
            labels = labels.to("cuda")

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute macro F1-score
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")


def predict_image(image_path, model, class_names, device="cpu"):
    """
    Predicts the class of a single image using the given model.

    Args:
        image_path (str): Path to the image.
        model (torch.nn.Module): Trained model.
        class_names (list): List of class name strings.
        device (str): Device to run inference on ("cpu" or "cuda").

    Returns:
        predicted_class_name (str), predicted_class_index (int), probability (float)
    """
    # Define the same transform used for validation
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        prob = probabilities[0, pred_idx].item()

    return class_names[pred_idx], pred_idx, prob


if __name__ == "__main__":
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    val_dataset = RiceDataset(
        "data/train_images",
        "data/meta_train.csv",
        "variety",
        split="val",
        transform=val_transform,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    model = CBAMResNet18(num_classes=10, weights="DEFAULT")
    model.load_state_dict(torch.load("checkpoints/final.pt", weights_only=True))
    model.to("cuda")

    print_macro_f1(val_dataloader, model)

    evaluate_and_plot_confusion_matrix(model, val_dataloader, val_dataset.classes)

    # Random sample
    import random

    random.seed(42)
    random_idx = random.randint(0, len(val_dataset) - 1)
    img, label = val_dataset[random_idx]

    input_tensor = img.unsqueeze(0).to("cuda")
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

    gradcam = GradCAM(model, model.layer4[-1])
    heatmap = gradcam.generate(input_tensor, class_idx=predicted_class)

    show_gradcam_on_image(
        img,
        heatmap,
        predicted_class=predicted_class,
        actual_class=label,
        class_names=val_dataset.classes,
    )

    show_random_prediction(
        val_dataset,
        model,
        class_names=val_dataset.classes,
        device="cuda",
        filter_classes=["Onthanel"],
    )

    print(
        predict_image(
            "data/train_images/bacterial_leaf_streak/100150.jpg",
            model,
            val_dataset.classes,
            device="cuda",
        )
    )
