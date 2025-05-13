import torch
import torch.nn as nn
import torch.nn.functional as F


# https://www.researchgate.net/publication/378679977_Enhancing_Agriculture_Crop_Classification_with_Deep_Learning


class PaperCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(PaperCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # Conv2D + ReLU
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # MaxPooling2D
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten layer
            nn.Linear(
                128 * 28 * 28, 128
            ),  # Adjust input size depending on input image size
            nn.ReLU(),
            nn.Linear(128, num_classes),  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
