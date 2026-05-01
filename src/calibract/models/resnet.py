# -------------------------
# resnet_base.py
# -------------------------
# Base ResNet-50 with optional adapter injection and encoder freezing

import copy
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from calibract.models.activations import BaseActivation, CustomActivationPlaceholder

from calibract.models.lora_layers import Conv2d as LoraConv, mark_only_lora_as_trainable  # LoRA-enhanced convolution



DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "merge_weights": True,
}


def conv3x3(
    in_planes,
    out_planes,
    stride=1,
    groups=1,
    dilation=1,
    is_lora=False,
    lora_config=None,
):
    if is_lora:
        return LoraConv(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
            r=lora_config.get("r", DEFAULT_LORA_CONFIG["r"]),
            lora_alpha=lora_config.get("lora_alpha", DEFAULT_LORA_CONFIG["lora_alpha"]),
            lora_dropout=lora_config.get("lora_dropout", DEFAULT_LORA_CONFIG["lora_dropout"]),
            merge_weights=lora_config.get("merge_weights", DEFAULT_LORA_CONFIG["merge_weights"]),
        )
    else:
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )


def conv1x1(in_planes, out_planes, stride=1, is_lora=False, lora_config=None):
    if is_lora:
        return LoraConv(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            bias=False,
            r=lora_config.get("r", DEFAULT_LORA_CONFIG["r"]),
            lora_alpha=lora_config.get("lora_alpha", DEFAULT_LORA_CONFIG["lora_alpha"]),
            lora_dropout=lora_config.get("lora_dropout", DEFAULT_LORA_CONFIG["lora_dropout"]),
            merge_weights=lora_config.get("merge_weights", DEFAULT_LORA_CONFIG["merge_weights"]),
        )
    else:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )


__all__ = [
    "resnet50_base",
    "ResNet",
    "Bottleneck",
    "initialize_lora_model",
    "initialize_basic_model",
    "freeze_non_activation_params",
]

model_urls = {"resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        is_lora=False,
        lora_config=None,
    ):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, is_lora=is_lora, lora_config=lora_config)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(
            planes, planes, stride=stride, is_lora=is_lora, lora_config=lora_config
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(
            planes, planes * self.expansion, is_lora=is_lora, lora_config=lora_config
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act1 = CustomActivationPlaceholder()
        self.act2 = CustomActivationPlaceholder()
        self.act3 = CustomActivationPlaceholder()
        self.downsample = downsample
        self.adapter = None
        self.adapter_in_channels = planes

    def forward(self, x):
        identity = x
        use_adapter_pre_conv2 = self.adapter is not None and self.conv2.stride == (1, 1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        if use_adapter_pre_conv2:
            adapter_out = self.adapter(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add adapter only if spatial dimensions match
        if self.adapter is not None:
            if use_adapter_pre_conv2:
                out = out + adapter_out
            else:
                out = out + self.adapter(out)

        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, num_classes=1000, is_lora=False, lora_config=None
    ):
        super().__init__()
        self.lora_config = lora_config
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = CustomActivationPlaceholder()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, 64, layers[0], is_lora=is_lora, lora_config=lora_config
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, is_lora=is_lora, lora_config=lora_config
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, is_lora=is_lora, lora_config=lora_config
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, is_lora=is_lora, lora_config=lora_config
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block, planes, blocks, stride=1, is_lora=False, lora_config=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes, planes * block.expansion, stride, is_lora=is_lora, lora_config=lora_config
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        for i in range(blocks):
            layers.append(
                block(
                    self.inplanes if i == 0 else planes * block.expansion,
                    planes,
                    stride if i == 0 else 1,
                    downsample if i == 0 else None,
                    is_lora=is_lora,
                    lora_config=lora_config,
                )
            )
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def set_custom_activation_map(self, activation_map, train_bn=False):
        # For each CustomActivationPlaceholder in the model, replace with config activation if present, else ReLU
        device = next(self.parameters()).device
        for module_name, module in self.named_modules():
            if isinstance(module, CustomActivationPlaceholder):
                parent = self
                attr_name = module_name
                # Traverse to parent module if nested
                if "." in module_name:
                    parts = module_name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    attr_name = parts[-1]
                # Replace the placeholder with the activation, and only unfreeze BN if activation is custom (not ReLU)
                if module_name in activation_map:
                    new_act = activation_map[module_name]
                else:
                    new_act = nn.ReLU()
                # Move custom activation to correct device if possible
                if hasattr(new_act, 'to'):
                    new_act = new_act.to(device)
                setattr(parent, attr_name, new_act)
                # Only unfreeze BatchNorm if the activation is NOT a plain nn.ReLU
                if train_bn and not (isinstance(new_act, nn.ReLU) and type(new_act) is nn.ReLU):
                    attrs = list(parent._modules.keys())
                    if attr_name in attrs:
                        idx = attrs.index(attr_name)
                        # Look for the previous attribute that is a BatchNorm layer
                        for prev_attr in reversed(attrs[:idx]):
                            prev_mod = getattr(parent, prev_attr)
                            if isinstance(prev_mod, nn.BatchNorm2d):
                                for param in prev_mod.parameters():
                                    param.requires_grad = True
                                # print(f"Unfroze BatchNorm layer '{prev_attr}' before activation '{attr_name}' in '{module_name}'")
                                break

    def load_weight_lora(self, state_dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        state_dict_tmp = copy.deepcopy(state_dict)

        remap = {}
        for key in list(state_dict_tmp.keys()):
            if "conv1.weight" in key and key != "conv1.weight":
                remap[key] = key.replace("conv1.weight", "conv1.conv.weight")
            elif "conv2.weight" in key:
                remap[key] = key.replace("conv2.weight", "conv2.conv.weight")
            elif "conv3.weight" in key:
                remap[key] = key.replace("conv3.weight", "conv3.conv.weight")
            elif "conv1.bias" in key:
                remap[key] = key.replace("conv1.bias", "conv1.conv.bias")
            elif "conv2.bias" in key:
                remap[key] = key.replace("conv2.bias", "conv2.conv.bias")
            elif "conv3.bias" in key:
                remap[key] = key.replace("conv3.bias", "conv3.conv.bias")
            elif "downsample.0.weight" in key:
                remap[key] = key.replace(
                    "downsample.0.weight", "downsample.0.conv.weight"
                )

        for old_key, new_key in remap.items():
            state_dict_tmp[new_key] = state_dict_tmp.pop(old_key)

        # Fill in missing keys from current model state (e.g., LoRA parameters)
        current_state = self.state_dict()
        for key in current_state:
            if key not in state_dict_tmp:
                state_dict_tmp[key] = current_state[key]

        self.load_state_dict(state_dict_tmp, strict=False)

def initialize_lora_model(num_classes, device, lora_config=None):
    model = resnet50_base(pretrained=True, num_classes=num_classes, is_lora=True, lora_config=lora_config)
    mark_only_lora_as_trainable(model, bias='none')
    model.to(device)
    return model

def freeze_non_activation_params(model):
    for name, module in model.named_modules():
        if isinstance(module, BaseActivation):
            # print(f"Leaving activation parameters unfrozen in: {name}")
            continue
        if isinstance(module, nn.Linear) and name == "fc":
            continue
        for _, param in module.named_parameters(recurse=False):
            param.requires_grad = False


def resnet50_base(
    pretrained=True, num_classes=1000, is_lora=False, lora_config=None, freeze=False, offline=True
):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        is_lora=is_lora,
        lora_config=lora_config,
    )
    if pretrained:
        if offline:
            try:
                state_dict = torch.load("/mnt/pretrained_model/resnet50-11ad3fa6.pth",weights_only=True) #switch to v2 for next set of experiments
            except FileNotFoundError:
                pretrained_dir = os.environ.get("CALIBRACT_PRETRAINED_DIR", "pretrained_model")
                local_path = os.path.join(pretrained_dir, "resnet50-11ad3fa6.pth")
                if os.path.exists(local_path):
                    print(f"Loading local pretrained weights from {local_path}")
                    state_dict = torch.load(local_path,weights_only=True)
                else:
                    print("Local pretrained weights not found; downloading ResNet-50 weights.")
                    state_dict = model_zoo.load_url(model_urls["resnet50"])
        else:
            state_dict = model_zoo.load_url(model_urls["resnet50"])
            print("Loaded pretrained model from URL:", model_urls["resnet50"])
        # Remove fc weights if shape mismatch is expected
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        if is_lora:
            model.load_weight_lora(state_dict)
        else:
            model.load_state_dict(state_dict, strict=False)
    if freeze:
        freeze_non_activation_params(model)
    return model


def initialize_basic_model(num_classes, device, freeze=False):
    model = resnet50_base(
        pretrained=True, num_classes=num_classes, is_lora=False, freeze=freeze
    )
    model.to(device)
    return model
