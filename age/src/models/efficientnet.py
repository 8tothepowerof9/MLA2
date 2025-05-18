import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchcam.methods import SmoothGradCAMpp

class EfficientNetAge(nn.Module):
    def __init__(self, use_gradcam=False):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)  # No pretrained weights as per your requirement
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_ftrs, 1)  # Output 1 continuous age value

        if use_gradcam:
            # Target the last Conv layer inside the features block
            self.gradcam = SmoothGradCAMpp(self, target_layer='backbone.features.7.1')  

    def forward(self, x):
        return self.backbone(x).view(-1)

    @property
    def cam_target_layer(self):
        return self.backbone.features[7][1]  # Last Conv layer block in EfficientNet-b0