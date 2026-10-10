#!/usr/bin/env bash
# Paper Table 3 (FLIR-aligned, 50 epochs): DyFCLT + baseline.
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/common.sh
for exp in dyfclt baseline; do
  launch_train configs/flir/$exp.yaml
  test_ckpts configs/flir/$exp.yaml
done
python tools/compare_with_paper.py --dataset flir \
  --run dyfclt=outputs/flir_dyfclt/test_metrics.json \
  --run baseline=outputs/flir_baseline/test_metrics.json --out outputs/compare_flir.md
