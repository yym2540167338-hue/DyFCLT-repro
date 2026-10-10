"""Convert RGBT-Tiny into the paired-COCO format (thermal annotations, as in the paper's Table 1).

RGBT-Tiny is gated: request access via the form linked from github.com/XinyiYing/RGBT-Tiny.
The public helper scripts of the dataset (github.com/AAAI989/RGBT-Tiny/codes/auxiliary_codes)
document this layout, which is what this script expects by default:

    ROOT/<sequence>/00/xxxxx.jpg + xxxxx.xml     visible frames + VOC-style boxes (name, id, bndbox)
    ROOT/<sequence>/01/xxxxx.jpg + xxxxx.xml     thermal frames + boxes

The benchmark split is 85 training / 30 test sequences. Pass the official split lists
(one sequence name per line) with --train-list/--test-list. If the release you receive
already contains COCO json files, use --coco-ir-json/--coco-split instead: the json is kept
and only the visible counterpart path is added (--ir-token/--rgb-token substitution).

Writes ROOT/annotations/{train,test}_paired.json. Run with --dry-run first to check counts.
"""
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET

CLASSES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]


def parse_xml(path):
    root = ET.parse(path).getroot()
    objs = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append({"name": name, "track_id": obj.findtext("id"), "box": [x1, y1, x2, y2]})
    size = root.find("size")
    wh = (int(float(size.findtext("width"))), int(float(size.findtext("height")))) if size is not None else None
    return objs, wh


def read_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def from_xml(args):
    seqs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d, args.ir_dir)))
    splits = {"train": read_list(args.train_list) if args.train_list else None,
              "test": read_list(args.test_list) if args.test_list else None}
    if splits["train"] is None or splits["test"] is None:
        raise SystemExit("please pass --train-list and --test-list (official 85/30 sequence split)")
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    name2id = {c["name"]: c["id"] for c in cats}
    unknown = {}
    for split, names in splits.items():
        images, anns = [], []
        for seq in names:
            if seq not in seqs:
                print(f"[warn] sequence {seq} not found under {args.root}")
                continue
            for xml in sorted(glob.glob(os.path.join(args.root, seq, args.ir_dir, "*.xml"))):
                stem = os.path.splitext(os.path.basename(xml))[0]
                ir_rel = f"{seq}/{args.ir_dir}/{stem}{args.ext}"
                rgb_rel = f"{seq}/{args.rgb_dir}/{stem}{args.ext}"
                objs, wh = parse_xml(xml)
                img_id = len(images) + 1
                w, h = wh or (640, 512)
                images.append({"id": img_id, "file_name": ir_rel, "file_name_rgb": rgb_rel, "width": w, "height": h,
                               "sequence": seq})
                for o in objs:
                    if o["name"] not in name2id:
                        unknown[o["name"]] = unknown.get(o["name"], 0) + 1
                        continue
                    x1, y1, x2, y2 = o["box"]
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": name2id[o["name"]],
                                 "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0, "track_id": o["track_id"]})
        print(f"{split}: {len(images)} frames, {len(anns)} boxes")
        if not args.dry_run:
            os.makedirs(os.path.join(args.root, "annotations"), exist_ok=True)
            json.dump({"images": images, "annotations": anns, "categories": cats},
                      open(os.path.join(args.root, "annotations", f"{split}_paired.json"), "w"))
    if unknown:
        print(f"[warn] skipped unknown class names: {unknown} (edit CLASSES if the release uses other spellings)")


def from_coco(args):
    d = json.load(open(args.coco_ir_json))
    for im in d["images"]:
        im["file_name_rgb"] = im["file_name"].replace(args.ir_token, args.rgb_token)
        if not os.path.isfile(os.path.join(args.root, im["file_name_rgb"])):
            raise FileNotFoundError(f"visible frame not found: {im['file_name_rgb']} (check --ir-token/--rgb-token)")
    for i, a in enumerate(d["annotations"], start=1):
        a["id"] = i
    print(f"{args.coco_split}: {len(d['images'])} frames, {len(d['annotations'])} boxes, categories "
          f"{[c['name'] for c in d['categories']]}")
    if not args.dry_run:
        os.makedirs(os.path.join(args.root, "annotations"), exist_ok=True)
        json.dump(d, open(os.path.join(args.root, "annotations", f"{args.coco_split}_paired.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/RGBT-Tiny")
    ap.add_argument("--train-list")
    ap.add_argument("--test-list")
    ap.add_argument("--ir-dir", default="01")
    ap.add_argument("--rgb-dir", default="00")
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--coco-ir-json", help="existing thermal COCO json (alternative input mode)")
    ap.add_argument("--coco-split", default="train")
    ap.add_argument("--ir-token", default="/01/")
    ap.add_argument("--rgb-token", default="/00/")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from_coco(a) if a.coco_ir_json else from_xml(a)
