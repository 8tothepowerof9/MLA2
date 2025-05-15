import torch
import torch.nn as nn
# class CNNAgeRegressor(nn.Module):
#     def __init__(self):
#         super(CNNAgeRegressor, self).__init__()

#         self.conv_block = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )

#         self.fc_block = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 28 * 28, 128),  #  input image size is 224x224
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 1)  # Regression output
#         )

#     def forward(self, x):
#         x = self.conv_block(x)
#         x = self.fc_block(x)
#         return x.view(-1)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAgeRegressor(nn.Module):
    def __init__(self):
        super(CNNAgeRegressor, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)  # Single regression output
        )
        


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.regressor(x)
        return x.view(-1)  # Regression output as 1D tensor per sample


