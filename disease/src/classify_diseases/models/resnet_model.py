import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet18Classifier, self).__init__()

        # Load ResNet18 architecture, but don't use pre-trained weights
        self.base_model = resnet18(weights=None)  # or pretrained=False for older versions

        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
