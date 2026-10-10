#!/usr/bin/env bash
# Paper Table 2 (RGBTDronePerson): DyFCLT + dual-branch RT-DETR baseline.  NGPU=1 bash scripts/run_droneperson.sh
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/common.sh
for exp in dyfclt baseline; do
  launch_train configs/droneperson/$exp.yaml
  test_ckpts configs/droneperson/$exp.yaml
done
python tools/compare_with_paper.py --dataset droneperson \
  --run dyfclt=outputs/droneperson_dyfclt/test_metrics.json \
  --run dyfclt_last=outputs/droneperson_dyfclt/test_metrics_last.json \
  --run baseline=outputs/droneperson_baseline/test_metrics.json --out outputs/compare_droneperson.md
