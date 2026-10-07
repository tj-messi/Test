# TGFA Temporary Reproduction Code

This is the temporary code repository used to reproduce the TGFA results reported in the paper. It contains the released inference and evaluation code, model configurations, checkpoints, and the cached text embeddings required to run the reported experiments.

## Installation

Run the commands below from the repository root with a Python environment that has a CUDA-compatible PyTorch installation:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

## Weights

The released weights are included in the repository:

```text
results/m3fd_hbb/m3fd_hbb.pth
results/vedai_obb/vedai_obb.pth
```

The matching model definitions are `configs/models/tgfa-n-ov.yaml` and `configs/models/tgfa-n-ov-obb.yaml`. The dataset configuration files are `configs/m3fd_hbb_inference.yml` and `configs/vedai_obb_inference.yml`.

## Inference

Run TGFA on one aligned visible/infrared image pair. The bundled examples can be used directly:

```bash
python inference/infer.py \
  --dataset m3fd_hbb \
  --rgb examples/inputs/m3fd/00001_rgb.png \
  --ir examples/inputs/m3fd/00001_ir.png \
  --output outputs/m3fd_example.json \
  --visualization outputs/m3fd_example.png \
  --device cuda:0
```

For VEDAI-OBB, use the corresponding pair:

```bash
python inference/infer.py \
  --dataset vedai_obb \
  --rgb examples/inputs/vedai/00000010_rgb.png \
  --ir examples/inputs/vedai/00000010_ir.png \
  --output outputs/vedai_example.json \
  --visualization outputs/vedai_example.png \
  --device cuda:0
```

For custom images, replace the `--rgb` and `--ir` paths. The output JSON contains the predicted classes, confidence scores, and boxes.

## Test

Run the complete validation evaluation used for the paper results.

### M3FD-HBB

```bash
python inference/eval.py \
  --dataset m3fd_hbb \
  --rgb-dir /path/to/M3FD_DEIM_MMT/images/val \
  --ir-dir /path/to/M3FD_DEIM_MMT/images_ir/val \
  --annotations /path/to/M3FD_DEIM_MMT/annotations/instances_val.json \
  --output outputs/m3fd_test.json \
  --device cuda:0 \
  --batch-size 16
```

### VEDAI-OBB

```bash
python inference/eval.py \
  --dataset vedai_obb \
  --rgb-dir /path/to/VEDAI/Vehicules1024 \
  --ir-dir /path/to/VEDAI/Vehicules1024 \
  --annotation-dir /path/to/VEDAI/Annotations1024 \
  --split-file /path/to/VEDAI/Annotations1024/fold01test.txt \
  --output outputs/vedai_test.json \
  --device cuda:0 \
  --batch-size 8
```

The M3FD test reports COCO-style `mAP50`, `mAP75`, and `mAP50-95`. The VEDAI test reports rotated-box `mAP50`, `mAP75`, and `mAP50-95`. Use `CUDA_VISIBLE_DEVICES=<index>` when a specific physical GPU should be selected.
