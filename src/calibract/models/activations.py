import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseActivation(nn.Module):
    """
    Base class for any activation function that wants to carry a named parameter group.
    This enables consistent logging, freezing, and identification across your model.
    """
    def __init__(self, group_name='default'):
        super().__init__()
        self.group_name = group_name
        self.clamped = 0

class CustomActivationPlaceholder(BaseActivation):
    def __init__(self, group_name='default'):
        super().__init__(group_name)
        self.act_fn = lambda x: x

    def set_activation(self, act_module):
        self.act_fn = act_module

    def forward(self, x):
        return self.act_fn(x)

class ChannelwiseActivation(BaseActivation):
    def __init__(self, activations_per_channel, group_name='default'):
        super().__init__(group_name)
        assert isinstance(activations_per_channel, list) or isinstance(activations_per_channel, nn.ModuleList), "activations_per_channel must be a list or ModuleList"
        self.activations = nn.ModuleList(activations_per_channel)

    def forward(self, x):
        assert x.shape[1] == len(self.activations), (
            f"Number of activations ({len(self.activations)}) must match number of channels ({x.shape[1]})"
        )
        outs = []
        for c in range(x.shape[1]):
            out = self.activations[c](x[:, c:c+1, :, :])
            outs.append(out)
        return torch.cat(outs, dim=1)

class KGActivationLaplacian(BaseActivation):
    def __init__(self, k=2, group_name='default'):
        super().__init__(group_name)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.k = k
        self.clamped = 0

        # Define Laplacian kernel
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, y):
        # Ensure kernel is on the same device as input
        laplacian_kernel = self.laplacian_kernel.to(y.device).expand(y.shape[1], 1, 3, 3)
        lap = F.conv2d(y, laplacian_kernel, padding=1, groups=y.shape[1])
        out = y + self.alpha * lap + self.beta * y + self.gamma * (y ** self.k)
        # Clamp output to prevent NaNs/Infs
        clamped_mask = (out.isnan() | out.isinf() | (out > 1e6) | (out < -1e6))
        num_clamped = clamped_mask.sum().item()
        if num_clamped > 0:
            self.clamped += num_clamped
            print(f"[CLAMPED] {num_clamped} elements clamped in KGActivationLaplacian (group: {self.group_name})")
        out = torch.clamp(out, -1e6, 1e6)

        return out


class KGActivationGeneral(BaseActivation):
    def __init__(self, group_name='default'):
        super().__init__(group_name)
        # Initialize parameters as learnable scalars
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.alpha * torch.cos(self.beta * x) + self.gamma * torch.sin(self.beta * x)

class PReLUActivation(BaseActivation):
    def __init__(self, num_parameters=1, init=0.25, group_name='default'):
        """
        num_parameters: 1 for shared across all channels, or C for per-channel learnable.
        """
        super().__init__(group_name)
        self.prelu = nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        return self.prelu(x)

class SwishFixed(BaseActivation):
    def __init__(self, group_name='default'):
        super().__init__(group_name)

    def forward(self, x):
        return x * torch.sigmoid(x)

class SwishLearnable(BaseActivation):
    def __init__(self, group_name='default'):
        super().__init__(group_name)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


channel_map = {
    # Top conv1 activation
    'activation': 64,

    # === Layer 1 ===
    'layer1.0.act1': 64,
    'layer1.0.act2': 64,
    'layer1.0.act3': 256,

    'layer1.1.act1': 64,
    'layer1.1.act2': 64,
    'layer1.1.act3': 256,

    'layer1.2.act1': 64,
    'layer1.2.act2': 64,
    'layer1.2.act3': 256,

    # === Layer 2 ===
    'layer2.0.act1': 128,
    'layer2.0.act2': 128,
    'layer2.0.act3': 512,

    'layer2.1.act1': 128,
    'layer2.1.act2': 128,
    'layer2.1.act3': 512,

    'layer2.2.act1': 128,
    'layer2.2.act2': 128,
    'layer2.2.act3': 512,

    'layer2.3.act1': 128,
    'layer2.3.act2': 128,
    'layer2.3.act3': 512,

    # === Layer 3 ===
    'layer3.0.act1': 256,
    'layer3.0.act2': 256,
    'layer3.0.act3': 1024,

    'layer3.1.act1': 256,
    'layer3.1.act2': 256,
    'layer3.1.act3': 1024,

    'layer3.2.act1': 256,
    'layer3.2.act2': 256,
    'layer3.2.act3': 1024,

    'layer3.3.act1': 256,
    'layer3.3.act2': 256,
    'layer3.3.act3': 1024,

    'layer3.4.act1': 256,
    'layer3.4.act2': 256,
    'layer3.4.act3': 1024,

    'layer3.5.act1': 256,
    'layer3.5.act2': 256,
    'layer3.5.act3': 1024,

    # === Layer 4 ===
    'layer4.0.act1': 512,
    'layer4.0.act2': 512,
    'layer4.0.act3': 2048,

    'layer4.1.act1': 512,
    'layer4.1.act2': 512,
    'layer4.1.act3': 2048,

    'layer4.2.act1': 512,
    'layer4.2.act2': 512,
    'layer4.2.act3': 2048,
}
