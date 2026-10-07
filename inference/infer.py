#!/usr/bin/env python3
"""RGB + infrared inference for the released checkpoints."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


DATASETS = {
    "m3fd_hbb": {
        "config": "configs/m3fd_hbb_inference.yml",
        "weights": "results/m3fd_hbb/m3fd_hbb.pth",
        "size": 640,
        "classes": ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"],
        "box_format": "xyxy",
    },
    "vedai_obb": {
        "config": "configs/vedai_obb_inference.yml",
        "weights": "results/vedai_obb/vedai_obb.pth",
        "size": 1024,
        "classes": [
            "car", "truck", "pickup", "tractor", "camping car", "boat",
            "motorcycle", "bus", "van", "other vehicle", "plane",
        ],
        "box_format": "cxcywh_angle_radians",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    return parser.parse_args()


COLORS = [
    "#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#9b5de5", "#00b4d8",
    "#ff006e", "#588157", "#ffbe0b", "#6d6875", "#3a86ff",
]


def oriented_corners(box):
    import math

    cx, cy, width, height, angle = box
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                 (width / 2, height / 2), (-width / 2, height / 2)):
        points.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a))
    return points


def save_visualization(rgb_path, detections, box_format, output_path):
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 300))
    for detection in detections:
        color = COLORS[detection["class_id"] % len(COLORS)]
        box = detection["box"]
        if box_format == "xyxy":
            draw.rectangle(box, outline=color, width=line_width)
            label_position = (box[0], max(0, box[1] - 12))
        else:
            points = oriented_corners(box)
            draw.line(points + [points[0]], fill=color, width=line_width, joint="curve")
            label_position = min(points, key=lambda point: point[1])
        label = f'{detection["class_name"]} {detection["score"]:.2f}'
        bounds = draw.textbbox(label_position, label, font=font, stroke_width=1)
        draw.rectangle(bounds, fill=color)
        draw.text(label_position, label, fill="white", font=font, stroke_width=1, stroke_fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def load_rgb(path, size):
    image = Image.open(path).convert("RGB")
    # Postprocessors consume sizes in (width, height) order.
    original_size = (image.width, image.height)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(__import__("numpy").array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), original_size


def load_ir(path, size):
    image = Image.open(path).convert("F").resize((size, size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(__import__("numpy").array(image, dtype="float32", copy=True)).unsqueeze(0)
    lo, hi = tensor.amin(), tensor.amax()
    tensor = (tensor - lo) / (hi - lo).clamp_min(1e-6)
    return tensor.unsqueeze(0)


def main():
    args = parse_args()
    package_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root))
    os.chdir(package_root)

    import engine  
    from engine.core import YAMLConfig

    spec = DATASETS[args.dataset]
    config_path = package_root / spec["config"]
    weight_path = package_root / spec["weights"]
    device = torch.device(args.device)

    cfg = YAMLConfig(str(config_path))
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"model"}:
        raise RuntimeError("Expected a released checkpoint with a single model key")
    model.load_state_dict(checkpoint["model"], strict=True)

    rgb, original_size = load_rgb(args.rgb, spec["size"])
    ir = load_ir(args.ir, spec["size"])
    sample = {"rgb": rgb.to(device), "npy": ir.to(device)}
    target_sizes = torch.tensor([original_size], dtype=torch.float32, device=device)

    with torch.inference_mode():
        outputs = model(sample)
        result = postprocessor(outputs, target_sizes)[0]

    keep = result["scores"] >= args.score_threshold
    labels = result["labels"][keep].cpu().tolist()
    scores = result["scores"][keep].cpu().tolist()
    boxes = result["boxes"][keep].cpu().tolist()
    detections = [
        {
            "class_id": int(label),
            "class_name": spec["classes"][int(label)],
            "score": float(score),
            "box": [float(value) for value in box],
        }
        for label, score, box in zip(labels, scores, boxes)
    ]
    payload = {
        "dataset": args.dataset,
        "rgb": str(args.rgb),
        "ir": str(args.ir),
        "image_size_hw": [int(original_size[1]), int(original_size[0])],
        "box_format": spec["box_format"],
        "score_threshold": args.score_threshold,
        "detections": detections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(detections)} detections to {args.output}")
    if args.visualization:
        save_visualization(args.rgb, detections, spec["box_format"], args.visualization)
        print(f"Wrote visualization to {args.visualization}")


if __name__ == "__main__":
    main()
