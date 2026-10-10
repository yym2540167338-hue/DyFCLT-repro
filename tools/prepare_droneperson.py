"""Convert the official RGBTDronePerson release into the paired-COCO format.

Input  (author Google Drive / Baidu release, unzipped):
    ROOT/train/{thermal,visible,annotation}/xxxxx.{jpg,xml}
    ROOT/val/{thermal,visible,annotation}/...
    ROOT/train_thermal.json, ROOT/val_thermal.json      (thermal COCO annotations)
Output:
    ROOT/annotations/{train,val}_paired.json
      images[i].file_name     = "<split>/thermal/xxxxx.jpg"   (annotated modality)
      images[i].file_name_rgb = "<split>/visible/xxxxx.jpg"
Fixes applied (both harmless for the official protocol):
  * annotation ids are re-indexed from 1 (pycocotools treats id 0 as "unmatched"),
  * width/height are taken from the image files (3 train entries say 630x500, files are 640x512).
The 'uncertain' category is kept in the file; configs train/evaluate person/rider/crowd only,
exactly like the official QFDet code (CLASSES = ('person', 'rider', 'crowd')).
"""
import argparse
import json
import os

from PIL import Image


def convert(root, split, src_json):
    d = json.load(open(os.path.join(root, src_json)))
    images = []
    for im in d["images"]:
        ir_rel = f"{split}/thermal/{im['file_name']}"
        rgb_rel = f"{split}/visible/{im['file_name']}"
        for rel in (ir_rel, rgb_rel):
            if not os.path.isfile(os.path.join(root, rel)):
                raise FileNotFoundError(os.path.join(root, rel))
        with Image.open(os.path.join(root, ir_rel)) as img:
            w, h = img.size
        images.append({"id": im["id"], "file_name": ir_rel, "file_name_rgb": rgb_rel, "width": w, "height": h})
    anns = []
    for i, a in enumerate(sorted(d["annotations"], key=lambda a: a["id"]), start=1):
        a = {k: v for k, v in a.items() if k != "segmentation"}
        a["id"] = i
        anns.append(a)
    out = {"images": images, "annotations": anns, "categories": d["categories"]}
    os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
    dst = os.path.join(root, "annotations", f"{split}_paired.json")
    json.dump(out, open(dst, "w"))
    print(f"{split}: {len(images)} images, {len(anns)} boxes -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/RGBTDronePerson")
    args = ap.parse_args()
    convert(args.root, "train", "train_thermal.json")
    convert(args.root, "val", "val_thermal.json")
