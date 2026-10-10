"""Dynamic Frequency-Band Decoupled Cross-Modal Attention (DFCA), paper Sec. 3.1.

For the IR branch the query comes from the visible feature and key/value come
from the infrared feature (Fig. 2(b)); the visible branch mirrors this.

    q,k,v      = DWConv3x3(Conv1x1(.))                             (Fig. 2(b))
    F_{m,b}    = M_b * FFT(F_m),  m in {q,k,v}                      (Eq. 1)
    M_b(u,v)   = 1[k_b <= sqrt(u^2+v^2) < k_{b+1}]                  (Eq. 2)
    A_b        = IFFT(F_{q,b} * conj(F_{k,b}))                      (Eq. 3)
    R_b        = sigmoid(Conv3x3(A_b)) * IFFT(F_{v,b})              (Eq. 4)
    F~         = Proj(LN(sum_b R_b))                                (Eq. 5)

Inner boundaries {k_b} are cumulative softplus increments initialised with the
octave scheme {0, 1/8, 1/4, 1/2} for B=3 (paper Sec. 3.1). Details the paper
does not specify are marked INFERRED and listed in REPRODUCTION_NOTES.md.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d


def _no_autocast(device_type):
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


class FrequencyBandDecomposition(nn.Module):
    """Radial frequency band masks with learnable, monotonically ordered boundaries.

    mask_mode:
      * ``soft`` (default, INFERRED): telescoping sigmoids, transition width =
        ``temperature`` frequency-grid steps. Needed for the boundaries to receive
        gradients and consistent with the Table 5 gap between "Query only" and
        "Query&Key" (with exactly binary masks these two variants are identical).
      * ``ste``: exactly binary masks of Eq. 2 in the forward pass, soft-mask
        gradients in the backward pass (straight-through).
      * ``hard``: exactly binary masks, boundaries receive no gradient.
    The top band is closed (it also holds the corner frequencies whose radius
    exceeds 1/2), so the bands always partition the full spectrum (INFERRED).
    """

    def __init__(self, num_bands=3, learnable=True, mask_mode="soft", temperature=1.0):
        super().__init__()
        assert num_bands >= 1
        self.num_bands = num_bands
        self.mask_mode = mask_mode
        self.temperature = temperature
        if num_bands > 1:
            inner = [0.5 / 2 ** (num_bands - b) for b in range(1, num_bands)]  # B=3 -> [1/8, 1/4]
            incr = torch.tensor([inner[0]] + [inner[i] - inner[i - 1] for i in range(1, len(inner))])
            raw = torch.log(torch.expm1(incr))  # inverse softplus
            if learnable:
                self.band_raw_increments = nn.Parameter(raw)
            else:
                self.register_buffer("band_raw_increments", raw)

    def boundaries(self):
        if self.num_bands == 1:
            return None
        return torch.cumsum(F.softplus(self.band_raw_increments), dim=0)

    def forward(self, h, w, device):
        """Masks of shape (num_bands, h, w // 2 + 1) on the rfft2 grid."""
        fy = torch.fft.fftfreq(h, device=device)
        fx = torch.fft.rfftfreq(w, device=device)
        radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        if self.num_bands == 1:
            return torch.ones(1, *radius.shape, device=device)
        k = self.boundaries().to(device)
        tau = self.temperature / min(h, w)
        diff = radius[None] - k[:, None, None]
        soft = torch.sigmoid(diff / tau)
        if self.mask_mode == "soft":
            above = soft
        elif self.mask_mode == "ste":
            above = (diff >= 0).float() + soft - soft.detach()
        elif self.mask_mode == "hard":
            above = (diff >= 0).float()
        else:
            raise ValueError(self.mask_mode)
        upper = torch.cat([torch.ones_like(above[:1]), above], 0)  # r >= k_b   (k_0 = 0)
        lower = torch.cat([above, torch.zeros_like(above[:1])], 0)  # r >= k_b+1 (k_B = inf)
        return upper - lower


class DFCA(nn.Module):
    def __init__(self, dim=256, num_bands=3, learnable_bands=True, decouple="qkv", mask_mode="soft",
                 band_temperature=1.0, mod_kernel=3, mod_groups=1):
        super().__init__()
        assert decouple in ("none", "q", "qk", "kv", "qkv"), decouple

        def qkv_proj():
            return nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 3, padding=1, groups=dim))

        self.q_proj, self.k_proj, self.v_proj = qkv_proj(), qkv_proj(), qkv_proj()
        self.decouple = decouple
        n_bands = 1 if decouple == "none" else num_bands
        self.fbd = FrequencyBandDecomposition(n_bands, learnable_bands, mask_mode, band_temperature)
        # one band-specific 3x3 modulation conv per band (Eq. 4); full conv by default (INFERRED)
        self.mod_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, mod_kernel, padding=mod_kernel // 2, groups=mod_groups) for _ in range(n_bands)])
        self.norm = LayerNorm2d(dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x_query, x_own):
        """x_query: feature of the other modality (provides Q); x_own: this branch's feature (K, V)."""
        q, k, v = self.q_proj(x_query), self.k_proj(x_own), self.v_proj(x_own)
        _, _, h, w = q.shape
        dq, dk, dv = (c in self.decouple for c in "qkv")
        with _no_autocast(q.device.type):
            q32, k32, v32 = q.float(), k.float(), v.float()
            fq = torch.fft.rfft2(q32, norm="ortho")
            fk = torch.fft.rfft2(k32, norm="ortho")
            fv = torch.fft.rfft2(v32, norm="ortho") if dv else None
            masks = self.fbd(h, w, q.device)
            corr, values = [], []
            for b in range(masks.shape[0]):
                m = masks[b]
                fq_b = fq * m if dq else fq
                fk_b = fk * m if dk else fk
                corr.append(torch.fft.irfft2(fq_b * torch.conj(fk_b), s=(h, w), norm="ortho"))
                values.append(torch.fft.irfft2(fv * m, s=(h, w), norm="ortho") if dv else v32)
        out = 0
        for b, (a_b, v_b) in enumerate(zip(corr, values)):
            gate = torch.sigmoid(self.mod_convs[b](a_b.to(x_own.dtype)))
            out = out + gate * v_b.to(x_own.dtype)
        return self.proj(self.norm(out))
