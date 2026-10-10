"""COCO-style evaluation for the three benchmarks used by the paper.

* ``coco``       - standard pycocotools protocol (FLIR-aligned, Table 3: AP50, AP).
* ``safit``      - RGBT-Tiny protocol (Table 1). The paper's baseline numbers in Table 1
                   are copied from the SAFit-based Table III of the RGBT-Tiny paper
                   (thermal annotations), so detections are matched with
                   SAFit = s*IoU + (1-s)*NWD(C),  s = sigmoid(sqrt(A_gt)/C - 1),  C = 32,
                   NWD(C) = exp(-W2/C) on Gaussian boxes [cx, cy, w/2, h/2].
                   Size ranges: tiny [1,8^2), extremely small [8^2,16^2), small [16^2,32^2),
                   medium [32^2,96^2), large >= 96^2 (Table 1 caption).
* ``tinyperson`` - RGBTDronePerson protocol (Table 2), ported from the official QFDet code
                   (NNNNerd/mmdet-rgbtdroneperson): IoU thresholds {0.25, 0.5, 0.75},
                   maxDets 1000, ranges tiny [1,20^2], small [20^2,32^2], ..., and unmatched
                   detections that lie inside an ignored (e.g. out-of-range) GT box with
                   IoD >= threshold are ignored instead of counted as false positives.
"""
import contextlib
import copy
import io
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

PROTOCOLS = {
    "coco": dict(
        iou_thrs=np.linspace(0.5, 0.95, 10), max_dets=[1, 10, 100],
        area_rng=[[0, 1e10], [0, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e10]],
        area_lbl=["all", "small", "medium", "large"]),
    "safit": dict(
        iou_thrs=np.linspace(0.5, 0.95, 10), max_dets=[1, 10, 100, 1000],
        area_rng=[[0, 1e10], [1, 8 ** 2], [8 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e10]],
        area_lbl=["all", "tiny", "extremely_small", "small", "medium", "large"]),
    "tinyperson": dict(
        iou_thrs=np.array([0.25, 0.5, 0.75]), max_dets=[100, 300, 1000],
        area_rng=[[1, 1e10], [1, 20 ** 2], [1, 8 ** 2], [8 ** 2, 12 ** 2], [12 ** 2, 20 ** 2], [20 ** 2, 32 ** 2],
                  [32 ** 2, 1e10]],
        area_lbl=["all", "tiny", "tiny1", "tiny2", "tiny3", "small", "reasonable"]),
}


def _iod(dets_xywh, gts_xywh):
    """Intersection over detection area, (D, G)."""
    d = np.asarray(dets_xywh, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(gts_xywh, dtype=np.float64).reshape(-1, 4)
    dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.clip(np.minimum(dx2[:, None], gx2[None]) - np.maximum(d[:, 0:1], g[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(dy2[:, None], gy2[None]) - np.maximum(d[:, 1:2], g[None, :, 1]), 0, None)
    darea = np.where((d[:, 2] > 0) & (d[:, 3] > 0), d[:, 2] * d[:, 3], 0.0)
    return iw * ih / (darea[:, None] + 1e-12)


def safit_similarity(dets_xywh, gts_xywh, ious, c=32.0, nwd_anchor="corner"):
    """SAFit as in the official RGBT-Tiny code (AAAI989/RGBT-Tiny codes/loss/loss.py):
    sigmoid((sqrt(A)-C)/C) * IoU + (1 - sigmoid(.)) * exp(-||g_p - g_t||_2 / C),
    where the official helper ``xyxy2xyw2h2`` uses [x1, y1, w/2, h/2] ("corner");
    the RGBT-Tiny paper text describes [cx, cy, w/2, h/2] ("center")."""
    d = np.asarray(dets_xywh, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(gts_xywh, dtype=np.float64).reshape(-1, 4)
    if nwd_anchor == "center":
        dg = np.stack([d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2, d[:, 2] / 2, d[:, 3] / 2], 1)
        gg = np.stack([g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2, g[:, 2] / 2, g[:, 3] / 2], 1)
    else:
        dg = np.stack([d[:, 0], d[:, 1], d[:, 2] / 2, d[:, 3] / 2], 1)
        gg = np.stack([g[:, 0], g[:, 1], g[:, 2] / 2, g[:, 3] / 2], 1)
    w2 = np.sqrt(((dg[:, None, :] - gg[None, :, :]) ** 2).sum(-1))
    nwd = np.exp(-w2 / c)
    s = 1.0 / (1.0 + np.exp(-(np.sqrt(g[:, 2] * g[:, 3]) - c) / c))
    return s[None, :] * np.asarray(ious) + (1.0 - s[None, :]) * nwd


class ExtendedCOCOeval(COCOeval):
    def __init__(self, coco_gt, coco_dt, protocol="coco", safit_c=32.0, use_iod_for_ignore=None, nwd_anchor="corner"):
        super().__init__(coco_gt, coco_dt, "bbox")
        cfg = PROTOCOLS[protocol]
        self.protocol = protocol
        self.params.iouThrs = np.asarray(cfg["iou_thrs"], dtype=np.float64)
        self.params.maxDets = list(cfg["max_dets"])
        self.params.areaRng = [list(a) for a in cfg["area_rng"]]
        self.params.areaRngLbl = list(cfg["area_lbl"])
        self.safit_c = safit_c
        self.nwd_anchor = nwd_anchor
        self.use_iod_for_ignore = protocol == "tinyperson" if use_iod_for_ignore is None else use_iod_for_ignore

    def _prepare(self):
        p = self.params
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        for g in gts:
            # annotation dicts are shared across evaluations -> keep the original attribute
            g.setdefault("_orig_ignore", g.get("ignore", 0))
            if self.protocol == "tinyperson":  # QFDet: 'ignore' attr OR iscrowd OR 'uncertain' attr
                g["ignore"] = int(bool(g["_orig_ignore"]) or bool(g.get("iscrowd", 0)) or bool(g.get("uncertain", 0)))
            else:  # pycocotools semantics: only crowd boxes are ignored
                g["ignore"] = int(bool(g.get("iscrowd", 0)))
        self._gts, self._dts = defaultdict(list), defaultdict(list)
        for g in gts:
            self._gts[g["image_id"], g["category_id"]].append(g)
        for d in dts:
            self._dts[d["image_id"], d["category_id"]].append(d)
        self.evalImgs = defaultdict(list)
        self.eval = {}

    def computeIoU(self, imgId, catId):
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds][: p.maxDets[-1]]
        g = [x["bbox"] for x in gt]
        d = [x["bbox"] for x in dt]
        ious = mask_utils.iou(d, g, [int(o["iscrowd"]) for o in gt])
        if self.protocol == "safit" and len(d) and len(g):
            ious = safit_similarity(d, g, ious, self.safit_c, self.nwd_anchor)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        if not self.use_iod_for_ignore:
            return super().evaluateImg(imgId, catId, aRng, maxDet)
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return None
        for g in gt:
            g["_ignore"] = 1 if (g["ignore"] or g["area"] < aRng[0] or g["area"] > aRng[1]) else 0
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
        T, G, D = len(p.iouThrs), len(gt), len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        ignore_idx = np.array([i for i, g in enumerate(gt) if g["_ignore"]], dtype=np.int64)
        iod = _iod([x["bbox"] for x in dt], [gt[i]["bbox"] for i in ignore_idx]) if len(ignore_idx) and D else None
        if len(ious) != 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        if ious[dind, gind] < iou:
                            continue
                        iou = ious[dind, gind]
                        m = gind
                    if m == -1:
                        if iod is None:
                            continue
                        j = int(np.argmax(iod[dind]))
                        if iod[dind, j] < iou:
                            continue
                        m = int(ignore_idx[j])
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        return {"image_id": imgId, "category_id": catId, "aRng": aRng, "maxDet": maxDet,
                "dtIds": [d["id"] for d in dt], "gtIds": [g["id"] for g in gt], "dtMatches": dtm, "gtMatches": gtm,
                "dtScores": [d["score"] for d in dt], "gtIgnore": gtIg, "dtIgnore": dtIg}

    def stat(self, ap=True, iou_thr=None, area="all", max_det=None):
        p = self.params
        max_det = p.maxDets[-1] if max_det is None else max_det
        aind = [i for i, a in enumerate(p.areaRngLbl) if a == area]
        mind = [i for i, m in enumerate(p.maxDets) if m == max_det]
        s = self.eval["precision"] if ap else self.eval["recall"]
        if iou_thr is not None:
            s = s[np.where(np.abs(iou_thr - p.iouThrs) < 1e-6)[0]]
        s = s[:, :, :, aind, mind] if ap else s[:, :, aind, mind]
        valid = s[s > -1]
        return float(np.mean(valid)) if valid.size else -1.0


def summarize(ev, protocol):
    m = {}
    if protocol in ("coco", "safit"):
        for md in ([100] if protocol == "coco" else [100, 1000]):
            sfx = "" if md == 100 else f"@{md}"
            m["AP" + sfx] = ev.stat(True, None, "all", md)
            m["AP50" + sfx] = ev.stat(True, 0.5, "all", md)
            m["AP75" + sfx] = ev.stat(True, 0.75, "all", md)
            for lbl in ev.params.areaRngLbl[1:]:
                m[f"AP_{lbl}" + sfx] = ev.stat(True, None, lbl, md)
            m["AR" + sfx] = ev.stat(False, None, "all", md)
            for lbl in ev.params.areaRngLbl[1:]:
                m[f"AR_{lbl}" + sfx] = ev.stat(False, None, lbl, md)
    else:
        for thr in (0.25, 0.5, 0.75):
            for lbl in ev.params.areaRngLbl:
                key = f"AP{int(thr * 100)}" + ("" if lbl == "all" else f"_{lbl}")
                m[key] = ev.stat(True, thr, lbl, 1000)
        m["AR50"] = ev.stat(False, 0.5, "all", 1000)
    return {k: (round(v * 100, 3) if v >= 0 else v) for k, v in m.items()}


def evaluate(coco_gt, results, protocol, cat_ids=None, img_ids=None, safit_c=32.0, nwd_anchor="corner", verbose=False):
    """results: list of {'image_id','category_id','bbox'(xywh),'score'}. Returns metrics in percent."""
    if len(results) == 0:
        return {}
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        coco_dt = coco_gt.loadRes(copy.deepcopy(results))
        ev = ExtendedCOCOeval(coco_gt, coco_dt, protocol, safit_c=safit_c, nwd_anchor=nwd_anchor)
        if cat_ids is not None:
            ev.params.catIds = list(cat_ids)
        if img_ids is not None:
            ev.params.imgIds = sorted(img_ids)
        ev.evaluate()
        ev.accumulate()
    if verbose:
        print(buf.getvalue())
    out = summarize(ev, protocol)
    out["per_class_AP50" if protocol != "coco" else "per_class_AP"] = per_class(ev, protocol, coco_gt)
    return out


def per_class(ev, protocol, coco_gt):
    p = ev.params
    prec = ev.eval["precision"]  # T,R,K,A,M
    res = {}
    for k, cid in enumerate(p.catIds):
        t = np.where(np.abs(p.iouThrs - 0.5) < 1e-6)[0] if protocol != "coco" else slice(None)
        s = prec[t, :, k, 0, -1]
        s = s[s > -1]
        res[coco_gt.cats[cid]["name"]] = round(float(np.mean(s)) * 100, 3) if s.size else -1.0
    return res
