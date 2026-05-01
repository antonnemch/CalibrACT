# -------------------------
# conv_adapter_module.py
# -------------------------
# Defines ConvAdapter block for ResNet from the paper Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets
# -------------------------

from calibract.models.resnet import Bottleneck, resnet50_base
import torch.nn as nn

class ConvAdapter(nn.Module):
    """
    ConvAdapter is a lightweight, residual adapter module inserted into each ResNet bottleneck block.
    It consists of a 1x1 → 3x3 depthwise → 1x1 convolutional bottleneck structure.
    """
    def __init__(self, in_channels, reduction=4, kernel_size=3):
        super().__init__()
        hidden_dim = in_channels // reduction
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.adapter(x)

def add_conv_to_resnet(model, reduction=4):
    """
    Adds ConvAdapters to each Bottleneck block of a ResNet model in-place.
    """
    for module in model.modules():
        if isinstance(module, Bottleneck):
            module.adapter = ConvAdapter(in_channels=module.adapter_in_channels, reduction=reduction)
    return model

def freeze_conv_encoder(model):
    """
    Freezes all ResNet weights except for adapters and the final classification head.
    """
    for name, param in model.named_parameters():
        if 'adapter' not in name and 'fc' not in name:
            param.requires_grad = False
            if 'fc' in name:
                print("Leaving FC layer trainable:", name)
    return model

def initialize_conv_model(num_classes, device, reduction):
    model = resnet50_base(pretrained=True, num_classes=num_classes)
    add_conv_to_resnet(model, reduction=reduction)
    freeze_conv_encoder(model)
    model.to(device)
    return model
