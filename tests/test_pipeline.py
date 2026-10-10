"""Correctness checks for the pieces that silently corrupt numbers if wrong.

    python -m pytest -q tests            (or: python tests/test_pipeline.py)

Uses a synthetic COCO set, so it runs without any dataset. test_tinyperson_matches_official
cross-checks against the official QFDet evaluator (tests/reference/qfdet_cocoeval.py, Apache-2.0).
"""
import contextlib
import copy
import importlib.util
import io
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from dyfclt.evaluation.cocoeval_ext import PROTOCOLS, ExtendedCOCOeval, evaluate, safit_similarity  # noqa: E402
from dyfclt.models.dfca import DFCA, FrequencyBandDecomposition  # noqa: E402
from dyfclt.models.sse import guided_upsample  # noqa: E402


def synthetic_coco(n_img=30, seed=0):
    rng = random.Random(seed)
    images, anns = [], []
    for i in range(1, n_img + 1):
        images.append({"id": i, "file_name": f"{i}.jpg", "width": 640, "height": 512})
        for _ in range(rng.randint(0, 15)):
            w, h = rng.uniform(3, 60), rng.uniform(3, 60)
            x, y = rng.uniform(0, 640 - w), rng.uniform(0, 512 - h)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": rng.randint(0, 2), "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": 0, "ignore": 0})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns,
                    "categories": [{"id": c, "name": n} for c, n in enumerate(["person", "rider", "crowd"])]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def noisy_dets(coco, seed=1, jitter=2.0, n_fp=10):
    rng = random.Random(seed)
    dets = []
    for a in coco.dataset["annotations"]:
        if rng.random() < 0.85:
            x, y, w, h = a["bbox"]
            dets.append({"image_id": a["image_id"], "category_id": a["category_id"], "score": rng.random(),
                         "bbox": [x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter),
                                  max(1, w + rng.uniform(-jitter, jitter)), max(1, h + rng.uniform(-jitter, jitter))]})
    for img in coco.dataset["images"]:
        for _ in range(n_fp):
            w, h = rng.uniform(3, 40), rng.uniform(3, 40)
            dets.append({"image_id": img["id"], "category_id": rng.randint(0, 2), "score": rng.random() * 0.5,
                         "bbox": [rng.uniform(0, 600), rng.uniform(0, 480), w, h]})
    return dets


def test_perfect_detections_give_100():
    coco = synthetic_coco()
    gt_dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
               for a in coco.dataset["annotations"]]
    for protocol, key in (("coco", "AP50"), ("safit", "AP50"), ("tinyperson", "AP50")):
        m = evaluate(coco, gt_dets, protocol)
        assert abs(m[key] - 100.0) < 1e-6, (protocol, m[key])


def test_coco_protocol_matches_pycocotools():
    coco = synthetic_coco()
    dets = noisy_dets(coco)
    ours = evaluate(coco, dets, "coco")
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(copy.deepcopy(dets)), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    assert abs(ours["AP"] - round(ev.stats[0] * 100, 3)) < 1e-6
    assert abs(ours["AP50"] - round(ev.stats[1] * 100, 3)) < 1e-6
    assert abs(ours["AR"] - round(ev.stats[8] * 100, 3)) < 1e-6


def test_tinyperson_matches_official():
    ref_path = os.environ.get("QFDET_COCOEVAL") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "reference", "qfdet_cocoeval.py")
    if not ref_path or not os.path.isfile(ref_path):
        print("skip: set QFDET_COCOEVAL to the official QFDet cocoeval.py to run this check")
        return
    spec = importlib.util.spec_from_file_location("qfdet_cocoeval", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    coco = synthetic_coco(n_img=40, seed=3)
    dets = noisy_dets(coco, seed=4, jitter=3.0, n_fp=20)
    ours = evaluate(coco, dets, "tinyperson")
    with contextlib.redirect_stdout(io.StringIO()):
        ref.Params.EVAL_STRANDARD = "tiny"
        ev = ref.COCOeval(coco, coco.loadRes(copy.deepcopy(dets)), "bbox", ignore_uncertain=True, use_iod_for_ignore=True)
        ev.params.catIds = [0, 1, 2]
        ev.params.maxDets = [100, 300, 1000]
        ev.params.iouThrs = np.linspace(0.25, 0.75, 3)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    # official stats order: [AP25 x 7 ranges, AP50 x 7, AP75 x 7, AR...]
    names = PROTOCOLS["tinyperson"]["area_lbl"]
    for t_i, thr in enumerate((25, 50, 75)):
        for a_i, lbl in enumerate(names):
            key = f"AP{thr}" + ("" if lbl == "all" else f"_{lbl}")
            want = ev.stats[t_i * len(names) + a_i]
            want = round(want * 100, 3) if want >= 0 else want
            assert abs(ours[key] - want) < 1e-6, (key, ours[key], want)
    print("tinyperson protocol identical to the official QFDet evaluator:",
          {k: ours[k] for k in ("AP50", "AP50_tiny", "AP50_small")})


def test_safit_formula():
    # identical boxes -> 1; for tiny boxes SAFit is far more tolerant than IoU to a 2px shift
    g = [[100, 100, 6, 6]]
    d_same = [[100, 100, 6, 6]]
    d_shift = [[102, 100, 6, 6]]
    iou_shift = 4 * 6 / (2 * 36 - 4 * 6)
    assert abs(safit_similarity(d_same, g, [[1.0]])[0, 0] - 1.0) < 1e-9
    s = safit_similarity(d_shift, g, [[iou_shift]])[0, 0]
    sig = 1 / (1 + np.exp(-(6 - 32) / 32))
    assert abs(s - (sig * iou_shift + (1 - sig) * np.exp(-2 / 32))) < 1e-9
    assert s > iou_shift
    # large objects: sigma -> 1, SAFit -> IoU
    big = safit_similarity([[0, 0, 400, 400]], [[10, 0, 400, 400]], [[0.95]])[0, 0]
    assert abs(big - 0.95) < 0.01


def test_band_masks_partition_and_init():
    for b in (1, 2, 3, 4):
        fbd = FrequencyBandDecomposition(b, learnable=True)
        m = fbd(40, 40, "cpu")
        assert m.shape == (b, 40, 21)
        assert torch.allclose(m.sum(0), torch.ones(40, 21), atol=1e-6)
    k = FrequencyBandDecomposition(3).boundaries()
    assert torch.allclose(k, torch.tensor([1 / 8, 1 / 4]), atol=1e-6)
    hard = FrequencyBandDecomposition(3, mask_mode="hard")(32, 32, "cpu")
    assert set(hard.unique().tolist()) <= {0.0, 1.0}


def test_band_boundaries_receive_gradient():
    torch.manual_seed(0)
    m = DFCA(32, num_bands=3)
    out = m(torch.randn(2, 32, 40, 40), torch.randn(2, 32, 40, 40))
    out.square().mean().backward()
    assert m.fbd.band_raw_increments.grad is not None and m.fbd.band_raw_increments.grad.abs().sum() > 0


def test_full_band_dfca_equals_gated_value():
    # with B=1 the FFT cross-correlation path must reduce to sigmoid(conv(q*k corr)) * v
    torch.manual_seed(0)
    m = DFCA(8, decouple="none")
    q_in, kv_in = torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16)
    q, k, v = m.q_proj(q_in), m.k_proj(kv_in), m.v_proj(kv_in)
    corr = torch.zeros_like(q)
    for dy in range(16):
        for dx in range(16):  # circular cross-correlation, scaled like ortho FFTs (1/sqrt(N))
            corr[..., dy, dx] = (torch.roll(k, shifts=(dy, dx), dims=(2, 3)) * q).sum((2, 3)) / 16.0
    want = m.proj(m.norm(torch.sigmoid(m.mod_convs[0](corr)) * v))
    assert torch.allclose(m(q_in, kv_in), want, atol=1e-4)


def test_guided_upsample_center_tap_is_nearest():
    x = torch.randn(2, 4, 5, 6)
    k = torch.zeros(2, 9, 10, 12)
    k[:, 4] = 1.0  # all weight on the centre tap
    y = guided_upsample(x, k)
    assert torch.allclose(y, torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"), atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
