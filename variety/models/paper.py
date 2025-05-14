import torch
import torch.nn as nn
from torchvision.models import vgg16


class PaperCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PaperCNN, self).__init__()
        # Load pretrained VGG16
        base_model = vgg16(pretrained=True)

        # Use all convolutional layers and pooling (feature extractor)
        self.features = base_model.features

        # Flatten + Dropout + Final Classification Layer
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Converts [B, 512, 7, 7] to [B, 25088]
            nn.Dropout(p=0.5),  # Dropout to reduce overfitting
            nn.Linear(25088, num_classes),  # Final classification layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
