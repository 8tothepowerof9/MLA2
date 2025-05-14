import torch
import torch.nn as nn
from torchvision.models import resnet34

class FineTunedResNet34(nn.Module):
    def __init__(self, num_classes=4, freeze_until_layer=6):
        """
        Args:
            num_classes (int): Number of output classes
            freeze_until_layer (int): How many children (blocks) to freeze (0 = freeze all, 9 = freeze none)
        """
        super(FineTunedResNet34, self).__init__()
        
        # Load pretrained resnet34
        base_model = resnet34(weights='IMAGENET1K_V1')
        
        # Freeze early layers
        children = list(base_model.children())
        for i, child in enumerate(children[:-2]):  # exclude avgpool + fc
            if i < freeze_until_layer:
                for param in child.parameters():
                    param.requires_grad = False
        
        # Rebuild model
        self.backbone = nn.Sequential(*children[:-2])  # all except avgpool and fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
