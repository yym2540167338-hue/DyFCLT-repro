"""Selective Smoothing Enhancement (SSE), paper Sec. 3.2 / Fig. 3.

Irrelevant Background Smoothing (IBS):
    M      = sigmoid(Conv(F~^l))                     mask, supervised by focal Tversky loss
    F_fg   = M * F~^l,  F_bg = (1 - M) * F~^l        (Eq. 6)
    F^_bg  = Conv3x3^C(Conv3x3^{C/r}(F_bg))          (Eq. 7)
    F_bgs  = F_fg + F^_bg                            (Eq. 8)
Foreground-Relevant Enhancement Fusion (FREF):
    V      = Conv3x3(F_bgs)  in R^{K^2 x 2H x 2W}     (Eq. 9)
    W      = softmax over the K^2 kernel taps         (Eq. 10)
    W~     = PixelUnshuffle(W) in R^{4K^2 x H x W}    (Eq. 11)
    Y_g    = sum_{(p,q)} W~_g(x,y) F~^{l+1}(x+p,y+q)  (Eq. 12)
    Y_up   = PixelShuffle(Y_1..Y_4)                   (Eq. 13)
    Y^     = Y_up + Upsample(F~^{l+1})                (Eq. 14)
    F_out  = Fusion(Y^, F_bgs)   (CSPBlock)           (Eq. 15)
With FREF disabled the block falls back to RT-DETR's CCFM top-down step
(lateral 1x1 conv -> nearest upsample -> concat -> CSPBlock).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormLayer, CSPRepLayer


def guided_upsample(x_high, kernels):
    """Content-aware x2 upsampling of ``x_high`` (B,C,H,W) with per-pixel kernels
    ``kernels`` (B,K*K,2H,2W) that are already softmax-normalised (Eqs. 11-13)."""
    b, c, h, w = x_high.shape
    k2 = kernels.shape[1]
    k = int(round(k2 ** 0.5))
    if kernels.shape[-2:] != (2 * h, 2 * w):
        kernels = F.interpolate(kernels, size=(2 * h, 2 * w), mode="nearest")
    neigh = F.unfold(x_high, kernel_size=k, padding=k // 2).view(b, c, k2, h, w)
    w_groups = F.pixel_unshuffle(kernels, 2).view(b, k2, 4, h, w)  # 4 sub-pixel groups
    y = torch.einsum("bckhw,bkghw->bcghw", neigh, w_groups).reshape(b, c * 4, h, w)
    return F.pixel_shuffle(y, 2)


class SSE(nn.Module):
    def __init__(self, dim=256, use_ibs=True, use_fref=True, reduction=4, up_kernel=5, mask_kernel=3,
                 act="silu", num_blocks=3, expansion=1.0):
        super().__init__()
        self.use_ibs = use_ibs
        self.use_fref = use_fref
        if use_ibs:
            self.mask_head = nn.Conv2d(dim, 1, mask_kernel, padding=mask_kernel // 2)
            self.smooth = nn.Sequential(
                ConvNormLayer(dim, dim // reduction, 3, 1, act=act),
                ConvNormLayer(dim // reduction, dim, 3, 1, act=act),
            )
            nn.init.constant_(self.mask_head.bias, -2.0)  # mostly-background prior (INFERRED)
        if use_fref:
            self.up_kernel = up_kernel
            self.kernel_conv = nn.Conv2d(dim, up_kernel ** 2, 3, padding=1)
        else:
            self.lateral = ConvNormLayer(dim, dim, 1, 1, act=act)
        self.fusion = CSPRepLayer(2 * dim, dim, num_blocks, expansion=expansion, act=act)

    def forward(self, x, x_high):
        """x: DFCA output at level l; x_high: already-processed feature of level l+1.

        Returns (F_out^l, mask_logits or None, feature that replaces level l+1)."""
        mask_logits = None
        if self.use_ibs:
            mask_logits = self.mask_head(x)
            m = torch.sigmoid(mask_logits)
            x = m * x + self.smooth((1 - m) * x)
        if self.use_fref:
            kernels = F.softmax(self.kernel_conv(x), dim=1)
            up = guided_upsample(x_high, kernels)
            if up.shape[-2:] != x.shape[-2:]:
                up = F.interpolate(up, size=x.shape[-2:], mode="bilinear", align_corners=False)
            up = up + F.interpolate(x_high, size=x.shape[-2:], mode="bilinear", align_corners=False)
            high_out = x_high
        else:
            high_out = self.lateral(x_high)
            up = F.interpolate(high_out, size=x.shape[-2:], mode="nearest")
        return self.fusion(torch.cat([up, x], dim=1)), mask_logits, high_out
