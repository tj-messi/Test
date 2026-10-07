#!/usr/bin/env python3
"""Evaluate a released checkpoint on M3FD or VEDAI validation data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


M3FD_CLASSES = ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"]
VEDAI_CLASSES = [
    "car", "truck", "pickup", "tractor", "camping car", "boat",
    "motorcycle", "bus", "van", "other vehicle", "plane",
]
VEDAI_CLASS_IDS = {1: 0, 2: 1, 4: 2, 5: 3, 7: 4, 8: 5, 9: 6,
                   10: 7, 11: 8, 23: 9, 31: 10}


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("m3fd_hbb", "vedai_obb"), required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path,
                        help="M3FD COCO annotation JSON.")
    parser.add_argument("--annotation-dir", type=Path,
                        help="VEDAI directory containing <image_id>.txt files.")
    parser.add_argument("--split-file", type=Path,
                        help="VEDAI split file, e.g. Annotations1024/fold01test.txt.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    return parser.parse_args()


def load_pair(rgb_path: Path, ir_path: Path, size: int):
    rgb_image = Image.open(rgb_path).convert("RGB")
    ir_image = Image.open(ir_path).convert("F")
    # Postprocessors consume sizes in (width, height) order.
    original_size = (rgb_image.width, rgb_image.height)
    rgb_image = rgb_image.resize((size, size), Image.Resampling.BILINEAR)
    ir_image = ir_image.resize((size, size), Image.Resampling.BILINEAR)
    rgb = torch.from_numpy(np.array(rgb_image, copy=True)).permute(2, 0, 1).float() / 255.0
    ir = torch.from_numpy(np.array(ir_image, dtype="float32", copy=True)).unsqueeze(0)
    lo, hi = ir.amin(), ir.amax()
    ir = (ir - lo) / (hi - lo).clamp_min(1e-6)
    return rgb, ir, original_size


def m3fd_pairs(rgb_dir: Path, ir_dir: Path, annotation_file: Path):
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    pairs = []
    for image in data["images"]:
        name = Path(image["file_name"]).name
        rgb = rgb_dir / name
        ir = ir_dir / name
        if not rgb.exists() or not ir.exists():
            raise FileNotFoundError(f"Missing aligned M3FD pair: {rgb} / {ir}")
        pairs.append((int(image["id"]), rgb, ir, image))
    return data, pairs


def vedai_pairs(rgb_dir: Path, ir_dir: Path, annotation_dir: Path, split_file: Path):
    if annotation_dir is None or split_file is None:
        raise ValueError("VEDAI requires --annotation-dir and --split-file")
    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    pairs = []
    for stem in ids:
        stem = Path(stem).stem
        rgb = rgb_dir / f"{stem}_co.png"
        ir = ir_dir / f"{stem}_ir.png"
        ann = annotation_dir / f"{stem}.txt"
        if not rgb.exists() or not ir.exists() or not ann.exists():
            raise FileNotFoundError(f"Missing VEDAI sample: {rgb}, {ir}, {ann}")
        pairs.append((int(stem), rgb, ir, ann))
    return pairs


def build_model(dataset: str, device: torch.device):
    package_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root))
    import engine  # noqa: F401
    from engine.core import YAMLConfig

    if dataset == "m3fd_hbb":
        config = package_root / "configs/m3fd_hbb_inference.yml"
        weights = package_root / "results/m3fd_hbb/m3fd_hbb.pth"
    else:
        config = package_root / "configs/vedai_obb_inference.yml"
        weights = package_root / "results/vedai_obb/vedai_obb.pth"
    cfg = YAMLConfig(str(config))
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"model"}:
        raise RuntimeError(f"Expected {{'model'}} checkpoint, got {set(checkpoint)}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, postprocessor


def run_inference(dataset, pairs, model, postprocessor, device, batch_size, threshold):
    size = 640 if dataset == "m3fd_hbb" else 1024
    predictions = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        rgb, ir, sizes = zip(*(load_pair(item[1], item[2], size) for item in batch))
        rgb = torch.stack(rgb).to(device)
        ir = torch.stack(ir).to(device)
        target_sizes = torch.tensor(sizes, dtype=torch.float32, device=device)
        with torch.inference_mode():
            outputs = model({"rgb": rgb, "npy": ir})
            results = postprocessor(outputs, target_sizes)
        for item, result in zip(batch, results):
            keep = result["scores"] >= threshold
            labels = result["labels"][keep].cpu().tolist()
            scores = result["scores"][keep].cpu().tolist()
            boxes = result["boxes"][keep].cpu().tolist()
            predictions.append((item[0], labels, scores, boxes))
        print(f"evaluated {min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    return predictions


def evaluate_m3fd(annotation_data, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO()
    coco.dataset = annotation_data
    coco.createIndex()
    rows = []
    for image_id, labels, scores, boxes in predictions:
        for label, score, box in zip(labels, scores, boxes):
            x1, y1, x2, y2 = box
            rows.append({"image_id": int(image_id), "category_id": int(label),
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    detections = coco.loadRes(rows) if rows else []
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"dataset": "m3fd_hbb", "images": len(annotation_data["images"]),
            "objects": len(annotation_data["annotations"]),
            "map50_95": float(evaluator.stats[0]), "map50": float(evaluator.stats[1]),
            "map75": float(evaluator.stats[2]), "predictions": len(rows)}


def read_vedai_box(path: Path, width: int, height: int):
    import cv2

    def canonical_min_area_rect(points):
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(points.astype(np.float32))
        if h > w:
            w, h = h, w
            angle_deg += 90.0
        angle_rad = math.radians(angle_deg)
        angle_rad = ((angle_rad + math.pi / 2) % math.pi) - math.pi / 2
        return float(cx), float(cy), float(w), float(h), float(angle_rad)

    labels = []
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) != 14:
            continue
        class_id = int(float(values[3]))
        if class_id not in VEDAI_CLASS_IDS:
            continue
        # VEDAI stores all four x coordinates followed by all four y
        # coordinates, rather than interleaved x/y pairs.
        xs = np.asarray([float(v) for v in values[6:10]], dtype=np.float32)
        ys = np.asarray([float(v) for v in values[10:14]], dtype=np.float32)
        if not ((0 <= xs).all() and (xs <= width).all()
                and (0 <= ys).all() and (ys <= height).all()):
            continue
        coords = np.stack([xs, ys], axis=1)
        cx, cy, w, h, angle = canonical_min_area_rect(coords)
        labels.append([VEDAI_CLASS_IDS[class_id], cx, cy, w, h, angle])
    return np.asarray(labels, dtype=np.float32).reshape(-1, 6)


def evaluate_vedai(predictions, pairs):
    from pycocoeval.yoloeval import ap_per_class, process_batch_obb

    ious = torch.linspace(0.5, 0.95, 10)
    stats = []
    gt_count = 0
    for item, pair in zip(predictions, pairs):
        image_id, labels, scores, boxes = item
        with Image.open(pair[1]) as image:
            width, height = image.size
        gt = read_vedai_box(pair[3], width, height)
        gt_count += len(gt)
        pred = np.asarray([box + [score, label] for label, score, box in zip(labels, scores, boxes)], dtype=np.float32)
        if pred.size == 0:
            stats.append((np.zeros((0, len(ious)), dtype=bool), np.zeros(0), np.zeros(0), gt[:, 0]))
            continue
        pred = pred.reshape(-1, 7)
        order = np.argsort(-pred[:, 5])
        pred = pred[order]
        correct = np.zeros((len(pred), len(ious)), dtype=bool)
        if len(gt):
            correct = process_batch_obb(
                torch.from_numpy(pred), torch.from_numpy(gt).float(), ious
            ).numpy()
        stats.append((correct, pred[:, 5], pred[:, 6], gt[:, 0]))

    if not stats:
        raise RuntimeError("No VEDAI samples were evaluated")
    tp, conf, pred_cls, target_cls = [np.concatenate(values, axis=0) for values in zip(*stats)]
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    _, _, _, _, _, ap, ap_class = ap_per_class(tp, conf, pred_cls, target_cls)
    per_class = {VEDAI_CLASSES[int(cls)]: float(ap[row, 0]) for row, cls in enumerate(ap_class)}
    return {"dataset": "vedai_obb", "images": len(pairs), "objects": gt_count,
            "map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()),
            "map75": float(ap[:, 5].mean()), "predictions": int(len(conf)),
            "per_class_map50": per_class}


def main():
    args = args_parser()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    device = torch.device(args.device)
    model, postprocessor = build_model(args.dataset, device)
    if args.dataset == "m3fd_hbb":
        if args.annotations is None:
            raise ValueError("M3FD requires --annotations")
        annotation_data, pairs = m3fd_pairs(args.rgb_dir, args.ir_dir, args.annotations)
    else:
        pairs = vedai_pairs(args.rgb_dir, args.ir_dir, args.annotation_dir, args.split_file)
        annotation_data = None
    predictions = run_inference(args.dataset, pairs, model, postprocessor, device,
                                args.batch_size, args.score_threshold)
    metrics = evaluate_m3fd(annotation_data, predictions) if args.dataset == "m3fd_hbb" else evaluate_vedai(predictions, pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
