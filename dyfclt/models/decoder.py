"""RT-DETR transformer decoder (deformable attention + IoU-aware query selection +
contrastive denoising), re-implemented after lyuwenyu/RT-DETR (Apache-2.0).
The paper keeps "the number of decoder layers and the number of queries ... the
same as in RT-DETR" (6 layers, 300 queries)."""
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .common import MLP, bias_init_with_prob, get_activation, inverse_sigmoid


def deformable_attention_core(value, spatial_shapes, sampling_locations, attention_weights):
    """Pure-PyTorch multi-scale deformable attention (works on CUDA, MPS and CPU)."""
    bs, _, n_head, c = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for lvl, (h, w) in enumerate(spatial_shapes):
        value_l = value_list[lvl].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        grid_l = sampling_grids[:, :, :, lvl].permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampled.append(F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False))
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(bs * n_head, 1, len_q, n_levels * n_points)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return out.reshape(bs, n_head * c, len_q).permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.num_levels, self.num_points = num_levels, num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dim // num_heads
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        grid_init *= torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.flatten())
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, spatial_shapes):
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        value = self.value_proj(value).reshape(bs, len_v, self.num_heads, self.head_dim)
        offsets = self.sampling_offsets(query).reshape(bs, len_q, self.num_heads, self.num_levels, self.num_points, 2)
        weights = self.attention_weights(query).reshape(bs, len_q, self.num_heads, self.num_levels * self.num_points)
        weights = F.softmax(weights, dim=-1).reshape(bs, len_q, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            normalizer = torch.tensor(spatial_shapes, device=query.device, dtype=query.dtype).flip([1])
            loc = reference_points.reshape(bs, len_q, 1, self.num_levels, 1, 2) + \
                offsets / normalizer.reshape(1, 1, 1, self.num_levels, 1, 2)
        else:
            loc = reference_points[:, :, None, :, None, :2] + \
                offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        return self.output_proj(deformable_attention_core(value, spatial_shapes, loc, weights))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0, activation="relu", n_levels=3, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, reference_points, memory, spatial_shapes, attn_mask=None, query_pos_embed=None):
        q = k = tgt if query_pos_embed is None else tgt + query_pos_embed
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        q = tgt if query_pos_embed is None else tgt + query_pos_embed
        tgt2 = self.cross_attn(q, reference_points, memory, spatial_shapes)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout4(tgt2))


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self, tgt, ref_points_unact, memory, spatial_shapes, bbox_head, score_head, query_pos_head, attn_mask=None):
        output = tgt
        dec_bboxes, dec_logits = [], []
        ref_points_detach = torch.sigmoid(ref_points_unact)
        ref_points = None
        for i, layer in enumerate(self.layers):
            query_pos = query_pos_head(ref_points_detach)
            output = layer(output, ref_points_detach.unsqueeze(2), memory, spatial_shapes, attn_mask, query_pos)
            inter_ref_bbox = torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            if self.training:
                dec_logits.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(inter_ref_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_logits.append(score_head[i](output))
                dec_bboxes.append(inter_ref_bbox)
                break
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox
        return torch.stack(dec_bboxes), torch.stack(dec_logits)


def get_contrastive_denoising_training_group(targets, num_classes, num_queries, class_embed, num_denoising=100,
                                             label_noise_ratio=0.5, box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None
    num_group = max(num_denoising // max_gt_num, 1)
    bs = len(num_gts)
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
    for i in range(bs):
        n = num_gts[i]
        if n > 0:
            input_query_class[i, :n] = targets[i]["labels"]
            input_query_bbox[i, :n] = targets[i]["boxes"]
            pad_gt_mask[i, :n] = 1
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = (1 - negative_gt_mask).squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(input_query_class, 0, num_classes)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part = rand_part * rand_sign
        known_bbox = (known_bbox + rand_part * diff).clip(min=0.0, max=1.0)
        input_query_bbox = inverse_sigmoid(box_xyxy_to_cxcywh(known_bbox))
    input_query_class = class_embed(input_query_class.long())

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True
    for i in range(num_group):
        s, e = max_gt_num * 2 * i, max_gt_num * 2 * (i + 1)
        if i == 0:
            attn_mask[s:e, e:num_denoising] = True
        if i == num_group - 1:
            attn_mask[s:e, :s] = True
        else:
            attn_mask[s:e, e:num_denoising] = True
            attn_mask[s:e, :s] = True
    dn_meta = {"dn_positive_idx": dn_positive_idx, "dn_num_group": num_group, "dn_num_split": [num_denoising, num_queries]}
    return input_query_class, input_query_bbox, attn_mask, dn_meta


class RTDETRDecoder(nn.Module):
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, feat_channels=(256, 256, 256),
                 feat_strides=(8, 16, 32), num_levels=3, num_decoder_points=4, nhead=8, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.0, activation="relu", num_denoising=100, label_noise_ratio=0.5,
                 box_noise_scale=1.0, learnt_init_query=False, eval_idx=-1, eps=1e-2, aux_loss=True):
        super().__init__()
        assert len(feat_channels) <= num_levels
        self.hidden_dim, self.nhead = hidden_dim, nhead
        self.feat_strides = list(feat_strides)
        self.num_levels, self.num_classes = num_levels, num_classes
        self.num_queries, self.eps = num_queries, eps
        self.num_decoder_layers, self.aux_loss = num_decoder_layers, aux_loss
        self.num_denoising, self.label_noise_ratio, self.box_noise_scale = num_denoising, label_noise_ratio, box_noise_scale

        self.input_proj = nn.ModuleList()
        for c in feat_channels:
            self.input_proj.append(nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(c, hidden_dim, 1, bias=False)), ("norm", nn.BatchNorm2d(hidden_dim))])))
        in_c = feat_channels[-1]
        for _ in range(num_levels - len(feat_channels)):
            self.input_proj.append(nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_c, hidden_dim, 3, 2, padding=1, bias=False)), ("norm", nn.BatchNorm2d(hidden_dim))])))
            in_c = hidden_dim

        layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, layer, num_decoder_layers, eval_idx)
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)])
        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _get_encoder_input(self, feats):
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]
        if self.num_levels > len(proj):
            for i in range(len(proj), self.num_levels):
                proj.append(self.input_proj[i](feats[-1] if i == len(feats) else proj[-1]))
        flat, shapes = [], []
        for f in proj:
            _, _, h, w = f.shape
            flat.append(f.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])
        return torch.cat(flat, 1), shapes

    def _generate_anchors(self, spatial_shapes, grid_size=0.05, dtype=torch.float32, device="cpu"):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            gy, gx = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij")
            grid_xy = (torch.stack([gx, gy], -1).unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.cat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        anchors = torch.cat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.full_like(anchors, float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        bs = memory.shape[0]
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_unact.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, 4))
        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.cat([denoising_bbox_unact, reference_points_unact], 1)
        enc_topk_logits = enc_outputs_class.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])).detach()
        if denoising_class is not None:
            target = torch.cat([denoising_class, target], 1)
        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0 and targets is not None:
            dn_class, dn_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets, self.num_classes, self.num_queries, self.denoising_class_embed, self.num_denoising,
                self.label_noise_ratio, self.box_noise_scale)
        else:
            dn_class, dn_bbox_unact, attn_mask, dn_meta = None, None, None, None
        target, init_ref_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
            memory, spatial_shapes, dn_class, dn_bbox_unact)
        out_bboxes, out_logits = self.decoder(target, init_ref_unact, memory, spatial_shapes, self.dec_bbox_head,
                                              self.dec_score_head, self.query_pos_head, attn_mask=attn_mask)
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        if self.training and dn_meta is None and self.num_denoising > 0:
            # batch without any GT box: keep the denoising embedding in the graph so that DDP
            # does not fail with "parameters that were not used in producing loss"
            out["pred_logits"] = out["pred_logits"] + 0.0 * self.denoising_class_embed.weight.sum()
        if self.training and self.aux_loss:
            out["aux_outputs"] = [{"pred_logits": a, "pred_boxes": b} for a, b in zip(out_logits[:-1], out_bboxes[:-1])]
            out["aux_outputs"].append({"pred_logits": enc_topk_logits, "pred_boxes": enc_topk_bboxes})
            if dn_meta is not None:
                out["dn_aux_outputs"] = [{"pred_logits": a, "pred_boxes": b} for a, b in zip(dn_out_logits, dn_out_bboxes)]
                out["dn_meta"] = dn_meta
        return out
