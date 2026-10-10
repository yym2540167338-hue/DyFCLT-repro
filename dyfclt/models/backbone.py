"""ResNet-50 backbones.

* ``presnet``: RT-DETR's PResNet (ResNet-vd) with the ImageNet weights released
  by the RT-DETR authors. This is the backbone of RT-DETR-R50, i.e. what you get
  when "adding a modality branch to RT-DETR" (paper Sec. 4.2).
* ``torchvision``: the original ResNet-50 [He et al.] from torchvision, in case the
  RT-DETR weights cannot be downloaded on the training server.
"""
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormLayer, freeze_batch_norm, get_activation

PRESNET_URLS = {
    18: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth",
    34: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth",
    50: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
    101: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth",
}
RESNET_CFG = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2b(self.branch2a(x))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__()
        stride1, stride2 = (stride, 1) if variant == "a" else (1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(ch_out, ch_out * self.expansion, 1, 1)
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ("conv", ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1)),
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2c(self.branch2b(self.branch2a(x)))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act="relu", variant="d"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(block(ch_in, ch_out, stride=2 if i == 0 and stage_num != 2 else 1,
                                     shortcut=i != 0, variant=variant, act=act))
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _load_state(url_or_path):
    if url_or_path and os.path.isfile(url_or_path):
        return torch.load(url_or_path, map_location="cpu")
    return torch.hub.load_state_dict_from_url(url_or_path, map_location="cpu")


class PResNet(nn.Module):
    def __init__(self, depth=50, variant="d", num_stages=4, return_idx=(1, 2, 3), act="relu",
                 freeze_at=0, freeze_norm=True, pretrained=True, pretrained_path=None):
        super().__init__()
        ch_in = 64
        if variant in ("c", "d"):
            conv_def = [[3, ch_in // 2, 3, 2, "conv1_1"], [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                        [ch_in // 2, ch_in, 3, 1, "conv1_3"]]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]
        self.conv1 = nn.Sequential(OrderedDict([
            (name, ConvNormLayer(ci, co, k, s, act=act)) for ci, co, k, s, name in conv_def]))
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock
        out_channels = [block.expansion * v for v in ch_out_list]
        out_strides = [4, 8, 16, 32]
        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            self.res_layers.append(Blocks(block, ch_in, ch_out_list[i], RESNET_CFG[depth][i], i + 2, act=act, variant=variant))
            ch_in = out_channels[i]
        self.return_idx = list(return_idx)
        self.out_channels = [out_channels[i] for i in self.return_idx]
        self.out_strides = [out_strides[i] for i in self.return_idx]

        if pretrained:
            state = _load_state(pretrained_path or PRESNET_URLS[depth])
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[PResNet{depth}] missing={len(missing)} unexpected={len(unexpected)}")
            print(f"[PResNet{depth}] loaded ImageNet weights from {pretrained_path or PRESNET_URLS[depth]}")
        if freeze_at >= 0:
            self._freeze(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze(self.res_layers[i])
        if freeze_norm:
            freeze_batch_norm(self)

    @staticmethod
    def _freeze(m):
        for p in m.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class TorchvisionResNet(nn.Module):
    def __init__(self, depth=50, return_idx=(1, 2, 3), freeze_at=0, freeze_norm=True, pretrained=True,
                 weights="IMAGENET1K_V1", pretrained_path=None):
        super().__init__()
        import torchvision

        net = getattr(torchvision.models, f"resnet{depth}")(weights=None)
        if pretrained:
            if pretrained_path:
                net.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
            else:
                w = torchvision.models.get_weight(f"ResNet{depth}_Weights.{weights}")
                net.load_state_dict(w.get_state_dict(progress=True))
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.res_layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.return_idx = list(return_idx)
        ch = [64, 128, 256, 512] if depth < 50 else [256, 512, 1024, 2048]
        self.out_channels = [ch[i] for i in self.return_idx]
        self.out_strides = [[4, 8, 16, 32][i] for i in self.return_idx]
        if freeze_at >= 0:
            for p in self.stem.parameters():
                p.requires_grad_(False)
            for i in range(min(freeze_at, 4)):
                for p in self.res_layers[i].parameters():
                    p.requires_grad_(False)
        if freeze_norm:
            freeze_batch_norm(self)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


def build_backbone(cfg):
    cfg = dict(cfg)
    kind = cfg.pop("type", "presnet")
    if kind == "presnet":
        return PResNet(**cfg)
    if kind == "torchvision":
        return TorchvisionResNet(**cfg)
    raise ValueError(f"unknown backbone type {kind}")
