import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision.models import efficientnet_b0
from torchcam.methods import SmoothGradCAMpp

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        channel_attn = self.sigmoid(self.channel_mlp(avg_out) + self.channel_mlp(max_out)).view(b, c, 1, 1)
        x = x * channel_attn

        avg_channel = torch.mean(x, dim=1, keepdim=True)
        max_channel, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial(torch.cat([avg_channel, max_channel], dim=1))
        return x * spatial_attn


class ResNetAgeWithCBAM(nn.Module):
    def __init__(self, use_gradcam=False):
        super().__init__()
        base = resnet34(weights=None)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)

        self.layer1 = base.layer1
        self.cbam1 = ChannelSpatialAttention(64)
        self.layer2 = base.layer2
        self.cbam2 = ChannelSpatialAttention(128)
        self.layer3 = base.layer3
        self.cbam3 = ChannelSpatialAttention(256)
        self.layer4 = base.layer4
        self.cbam4 = ChannelSpatialAttention(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

        if use_gradcam:
            from torchcam.methods import SmoothGradCAMpp
            self.gradcam = SmoothGradCAMpp(self, target_layer=self.layer4[1].conv2)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.cbam1(x)
        x = self.layer2(x)
        x = self.cbam2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.view(-1)

    @property
    def cam_target_layer(self):
        return self.layer4[1].conv2

class EfficientNetAgeWithCBAM(nn.Module):
    def __init__(self, use_gradcam=False):
        super().__init__()
        base = efficientnet_b0(weights=None)

        self.features = base.features

        # Apply CBAM after selected stages
        self.cbam_indices = [2, 4, 6]  # indices of blocks after which to apply CBAM
        self.cbams = nn.ModuleList([ChannelSpatialAttention(self.features[i][0].out_channels) for i in self.cbam_indices])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 1)  # 1280 is the default output channel of EfficientNet-B0

        if use_gradcam:
            self.gradcam = SmoothGradCAMpp(self, target_layer=self.features[6][0])

    def forward(self, x):
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.cbam_indices:
                idx = self.cbam_indices.index(i)
                x = self.cbams[idx](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.view(-1)

    @property
    def cam_target_layer(self):
        return self.features[6][0]