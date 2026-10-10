"""DyFCLT dual-branch encoder (paper Fig. 2(a)).

Per modality branch (IR shown, RGB symmetric):
    level L-1 (S5):  Self-Attn(F_ir, F_ir, F_ir)       RT-DETR AIFI layer
    levels < L-1:    DFCA(q <- F_vis, k,v <- F_ir)      -> SSE top-down (IBS + FREF)
Then per level the RGB/IR outputs are fused by a CSPBlock ("Fusion") and passed to the
RT-DETR decoder (concat + flatten).

Switches reproduce the ablations: use_dfca / use_ibs / use_fref (Table 4),
dfca.decouple (Table 5), dfca.num_bands / dfca.learnable_bands (Table 6).
``use_pan=True`` adds RT-DETR's bottom-up PAN path inside each branch, so that with
all DyFCLT modules disabled every branch is exactly an RT-DETR HybridEncoder
("dual-branch RT-DETR" baseline).
"""
import copy

import torch
import torch.nn as nn

from .common import ConvNormLayer, CSPRepLayer, get_activation
from .dfca import DFCA
from .sse import SSE


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.0, activation="gelu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, src, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = src if pos_embed is None else src + pos_embed
        src, _ = self.self_attn(q, k, value=src)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
    """Identical to RT-DETR (including its w/h meshgrid ordering)."""
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    pos_dim = embed_dim // 4
    omega = 1.0 / (temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]
    return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None]


class AIFI(nn.Module):
    """Attention-based intra-scale feature interaction on the top level (the 'Self-Attn' box)."""

    def __init__(self, hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.0, activation="gelu",
                 num_layers=1, pe_temperature=10000.0):
        super().__init__()
        layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.pe_temperature = pe_temperature

    def forward(self, x):
        b, c, h, w = x.shape
        src = x.flatten(2).permute(0, 2, 1)
        pos = build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src.device, src.dtype)
        for layer in self.layers:
            src = layer(src, pos_embed=pos)
        return src.permute(0, 2, 1).reshape(b, c, h, w).contiguous()


class Branch(nn.Module):
    """One modality branch of DyFCLT."""

    def __init__(self, in_channels, hidden_dim, aifi_cfg, use_dfca, dfca_cfg, use_ibs, use_fref, sse_cfg,
                 use_pan, act, num_blocks, expansion):
        super().__init__()
        n = len(in_channels)
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim)) for c in in_channels])
        self.aifi = AIFI(hidden_dim, **aifi_cfg)
        self.dfca = nn.ModuleList([DFCA(hidden_dim, **dfca_cfg) for _ in range(n - 1)]) if use_dfca else None
        # sse[i] handles level i (0 = highest resolution) using level i+1 as the higher-level input
        self.sse = nn.ModuleList([
            SSE(hidden_dim, use_ibs=use_ibs, use_fref=use_fref, act=act, num_blocks=num_blocks, expansion=expansion,
                **sse_cfg) for _ in range(n - 1)])
        self.use_pan = use_pan
        if use_pan:
            self.downsample_convs = nn.ModuleList([ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act) for _ in range(n - 1)])
            self.pan_blocks = nn.ModuleList([
                CSPRepLayer(2 * hidden_dim, hidden_dim, num_blocks, expansion=expansion, act=act) for _ in range(n - 1)])

    def project(self, feats):
        proj = [p(f) for p, f in zip(self.input_proj, feats)]
        proj[-1] = self.aifi(proj[-1])
        return proj

    def interact(self, own, other):
        """DFCA at every level except the top one (Fig. 2(a))."""
        if self.dfca is None:
            return list(own)
        return [self.dfca[i](other[i], own[i]) for i in range(len(own) - 1)] + [own[-1]]

    def top_down_and_pan(self, feats):
        n = len(feats)
        inner = list(feats)
        masks = []
        for i in range(n - 2, -1, -1):
            out, mask_logits, high = self.sse[i](inner[i], inner[i + 1])
            inner[i + 1] = high
            inner[i] = out
            if mask_logits is not None:
                masks.append(mask_logits)
        if not self.use_pan:
            return inner, masks
        outs = [inner[0]]
        for i in range(n - 1):
            down = self.downsample_convs[i](outs[-1])
            outs.append(self.pan_blocks[i](torch.cat([down, inner[i + 1]], dim=1)))
        return outs, masks


class DyFCLTEncoder(nn.Module):
    def __init__(self, in_channels=(512, 1024, 2048), hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.0,
                 enc_act="gelu", num_encoder_layers=1, pe_temperature=10000.0, act="silu", expansion=1.0,
                 depth_mult=1.0, use_dfca=True, use_ibs=True, use_fref=True, use_pan=False, dfca=None, sse=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = [hidden_dim] * len(in_channels)
        num_blocks = round(3 * depth_mult)
        aifi_cfg = dict(nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=enc_act,
                        num_layers=num_encoder_layers, pe_temperature=pe_temperature)
        args = (list(in_channels), hidden_dim, aifi_cfg, use_dfca, dict(dfca or {}), use_ibs, use_fref,
                dict(sse or {}), use_pan, act, num_blocks, expansion)
        self.ir = Branch(*args)
        self.vis = Branch(*args)
        self.fusion = nn.ModuleList([
            CSPRepLayer(2 * hidden_dim, hidden_dim, num_blocks, expansion=expansion, act=act) for _ in in_channels])

    def forward(self, feats_ir, feats_vis):
        p_ir = self.ir.project(feats_ir)
        p_vis = self.vis.project(feats_vis)
        d_ir = self.ir.interact(p_ir, p_vis)    # IR branch: q <- RGB, k/v <- IR
        d_vis = self.vis.interact(p_vis, p_ir)  # RGB branch: q <- IR, k/v <- RGB
        o_ir, m_ir = self.ir.top_down_and_pan(d_ir)
        o_vis, m_vis = self.vis.top_down_and_pan(d_vis)
        outs = [f(torch.cat([a, b], dim=1)) for f, a, b in zip(self.fusion, o_ir, o_vis)]
        return outs, m_ir + m_vis
