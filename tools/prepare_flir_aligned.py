"""Convert the "aligned" FLIR release of Zhang et al. (ICIP'20) into the paired-COCO format.

Download (4,129 train / 1,013 test pairs): the authors' link
https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view or the HF mirror
jsonhash/FLIR_aligned (aligned.zip). FLIR ADAS licence terms apply.

Expected layout after unzipping (VOC style; adjust the flags if your copy differs):
    ROOT/JPEGImages/FLIR_xxxxx_PreviewData.jpeg   thermal
    ROOT/JPEGImages/FLIR_xxxxx_RGB.jpg            visible
    ROOT/Annotations/FLIR_xxxxx_PreviewData.xml   boxes (thermal)
    ROOT/ImageSets/Main/{train,test}.txt (or align_train.txt / align_validation.txt)
Classes follow the common 3-class protocol: person, car, bicycle.
NOTE: written from the documented release structure; not run against the real files here.
"""
import argparse
import os
import json
import xml.etree.ElementTree as ET

CLASSES = ["person", "car", "bicycle"]


def main(args):
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    name2id = {c["name"]: c["id"] for c in cats}
    for split, list_file in (("train", args.train_list), ("test", args.test_list)):
        with open(os.path.join(args.root, list_file)) as f:
            stems = [l.strip().split()[0] for l in f if l.strip()]
        images, anns, skipped = [], [], {}
        for stem in stems:
            base = stem.replace(args.ir_suffix, "")
            ir_rel = f"{args.img_dir}/{base}{args.ir_suffix}{args.ir_ext}"
            rgb_rel = f"{args.img_dir}/{base}{args.rgb_suffix}{args.rgb_ext}"
            xml = os.path.join(args.root, args.ann_dir, f"{base}{args.ir_suffix}.xml")
            root = ET.parse(xml).getroot()
            size = root.find("size")
            w, h = int(size.findtext("width")), int(size.findtext("height"))
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": ir_rel, "file_name_rgb": rgb_rel, "width": w, "height": h})
            for obj in root.iter("object"):
                name = obj.findtext("name").strip().lower()
                if name not in name2id:
                    skipped[name] = skipped.get(name, 0) + 1
                    continue
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                if x2 <= x1 or y2 <= y1:
                    continue
                anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": name2id[name],
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        missing = [im["file_name_rgb"] for im in images[:20] if not os.path.isfile(os.path.join(args.root, im["file_name_rgb"]))]
        if missing:
            raise FileNotFoundError(f"visible frames not found, e.g. {missing[:3]}: check --rgb-suffix/--rgb-ext")
        print(f"{split}: {len(images)} pairs, {len(anns)} boxes, skipped classes {skipped}")
        os.makedirs(os.path.join(args.root, "annotations"), exist_ok=True)
        json.dump({"images": images, "annotations": anns, "categories": cats},
                  open(os.path.join(args.root, "annotations", f"{split}_paired.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/FLIR_aligned")
    ap.add_argument("--train-list", default="ImageSets/Main/train.txt")
    ap.add_argument("--test-list", default="ImageSets/Main/test.txt")
    ap.add_argument("--img-dir", default="JPEGImages")
    ap.add_argument("--ann-dir", default="Annotations")
    ap.add_argument("--ir-suffix", default="_PreviewData")
    ap.add_argument("--ir-ext", default=".jpeg")
    ap.add_argument("--rgb-suffix", default="_RGB")
    ap.add_argument("--rgb-ext", default=".jpg")
    main(ap.parse_args())
