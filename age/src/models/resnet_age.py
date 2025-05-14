import torch.nn as nn
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

class ResNetAge(nn.Module):
    def __init__(self, use_gradcam=False):
        super().__init__()
        self.backbone = resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)
        
        if use_gradcam:
            self.gradcam = SmoothGradCAMpp(self, target_layer='backbone.layer4.1.conv2')

    def forward(self, x):
        return self.backbone(x).view(-1)
    @property
    def cam_target_layer(self):
        return self.backbone.layer4[1].conv2  # for ResNetAge
