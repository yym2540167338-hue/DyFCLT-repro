"""Shared building blocks.

ConvNormLayer / RepVggBlock / CSPRepLayer / MLP / FrozenBatchNorm2d follow the
public RT-DETR implementation (lyuwenyu/RT-DETR, Apache-2.0), because the paper
builds its baseline by "adding a modality branch to RT-DETR" and calls the
RT-DETR fusion block a "CSPBlock".
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act):
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act == "silu":
        return nn.SiLU()
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    if act == "leaky_relu":
        return nn.LeakyReLU()
    if act in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"unknown activation {act}")


def bias_init_with_prob(prior_prob=0.01):
    return float(-math.log((1 - prior_prob) / prior_prob))


def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, padding=None, groups=1, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding=padding, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class CSPRepLayer(nn.Module):
    """RT-DETR's CSP block (RepC3). Used for every 'Fusion' block in DyFCLT."""

    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu"):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[RepVggBlock(hidden, hidden, act=act) for _ in range(num_blocks)])
        self.conv3 = ConvNormLayer(hidden, out_channels, 1, 1, bias=bias, act=act) if hidden != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) tensor."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen statistics and affine parameters (RT-DETR freeze_norm=True)."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        return x * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}"


def freeze_batch_norm(module):
    """Recursively replace nn.BatchNorm2d by FrozenBatchNorm2d (keeps loaded statistics)."""
    if isinstance(module, nn.BatchNorm2d):
        frozen = FrozenBatchNorm2d(module.num_features, module.eps)
        frozen.weight.copy_(module.weight.detach())
        frozen.bias.copy_(module.bias.detach())
        frozen.running_mean.copy_(module.running_mean)
        frozen.running_var.copy_(module.running_var)
        return frozen
    for name, child in module.named_children():
        new_child = freeze_batch_norm(child)
        if new_child is not child:
            setattr(module, name, new_child)
    return module
