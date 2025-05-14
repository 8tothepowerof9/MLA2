import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from torchvision.transforms.functional import to_pil_image
import traceback
import torch.nn.functional as F

class ModelVisualizer:
    def __init__(self, output_dir="age/checkpoints", model_name="model"):
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_metrics(self, log_path):
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")

        df = pd.read_csv(log_path)

        metrics = [
            ("train_loss", "val_loss"),
            ("train_mae", "val_mae"),
            ("train_r2", "val_r2")
        ]

        for metric_group in metrics:
            plt.figure()
            for metric in metric_group:
                if metric in df:
                    plt.plot(df[metric], label=metric)

            plt.title(" / ".join(metric_group))
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            name = f"{self.model_name}_" + "_".join(metric_group) + ".png"
            save_path = os.path.join(self.output_dir, name)
            plt.savefig(save_path)
            plt.close()

        print(f"Metric plots saved to {self.output_dir}")

    def plot_residuals(self, model, dataloader, device):
        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for images, target, _ in dataloader:
                images = images.to(device)
                target = target.float().to(device)
                outputs = model(images).view(-1)
                # denormalize the output to the original age range
                preds.extend((outputs * (82 - 45) + 45).cpu().numpy())
                targets.extend(target.cpu().numpy())

        residuals = [pred - true for pred, true in zip(preds, targets)]

        plt.figure(figsize=(8, 6))
        plt.scatter(targets, residuals, alpha=0.4, edgecolors='k')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Actual Age")
        plt.ylabel("Residual (Predicted - Actual)")
        plt.title("Residual Plot")
        plt.grid(True)

        save_path = os.path.join(self.output_dir, f"{self.model_name}_residual_plot.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Residual plot saved to {save_path}")
    
        
    def plot_gradcam(self, model, sample, device, csv_path=None):
        try:
            model.eval()
            model.to(device)

            image_tensor, actual_age, image_id = sample
            image_tensor = image_tensor.to(device).unsqueeze(0)
            image_tensor.requires_grad = True

            # Optional CSV override for actual_age if needed
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                match_row = df[df['image_id'] == image_id]
                if not match_row.empty:
                    actual_age = match_row['age'].values[0]

            activations = {}
            gradients = {}

            def forward_hook(module, input, output):
                activations["value"] = output

            def backward_hook(module, grad_in, grad_out):
                gradients["value"] = grad_out[0]

            target_layer = model.cam_target_layer
            handle_fwd = target_layer.register_forward_hook(forward_hook)
            handle_bwd = target_layer.register_full_backward_hook(backward_hook)

            output = model(image_tensor)
            model.zero_grad()
            output.backward()

            grads = gradients["value"]
            acts = activations["value"]
            pooled_grads = grads.mean(dim=(2, 3), keepdim=True)
            cam = (pooled_grads * acts).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu()

            image_np = image_tensor.squeeze().detach().cpu()
            image_pil = to_pil_image(image_np)

            plt.figure(figsize=(6, 6))
            plt.imshow(image_pil)
            plt.imshow(cam.numpy(), cmap="jet", alpha=0.4)
            plt.axis("off")

            # Denormalizing 
            AGE_MIN = 45.0
            AGE_MAX = 82.0
            predicted_age = output.item() * (AGE_MAX - AGE_MIN) + AGE_MIN

            title = f"Grad-CAM - {self.model_name} (Pred: {predicted_age:.1f})"
            if actual_age is not None:
                title += f" | Actual: {actual_age:.1f}"
            plt.title(title)

            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, f"{self.model_name}_gradcam_{image_id}")
            plt.savefig(save_path)
            plt.close()
            print(f"Grad-CAM saved to {save_path}")

            handle_fwd.remove()
            handle_bwd.remove()

        except Exception as e:
            print("[ERROR] GradCAM failed with exception:")
            traceback.print_exc()


def compute_regression_metrics(preds, targets):
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return mae, r2

def get_image_tensor_by_id(dataset, target_image_id: str):
    print(dataset)
    for image, target, image_id in dataset:
        if str(image_id).strip().lower() == str(target_image_id).strip().lower():
            return image, target
    raise ValueError(f"Image ID {target_image_id} not found in dataset")

