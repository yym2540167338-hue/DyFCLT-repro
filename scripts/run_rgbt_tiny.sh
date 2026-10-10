#!/usr/bin/env bash
# Paper Table 1 (RGBT-Tiny): DyFCLT + baseline.  ABLATIONS=1 also runs Tables 4-6 (11 extra trainings).
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/common.sh
for exp in dyfclt baseline; do
  launch_train configs/rgbt_tiny/$exp.yaml
  test_ckpts configs/rgbt_tiny/$exp.yaml
done
python tools/compare_with_paper.py --dataset rgbt_tiny \
  --run dyfclt=outputs/rgbt_tiny_dyfclt/test_metrics.json \
  --run baseline=outputs/rgbt_tiny_baseline/test_metrics.json --out outputs/compare_rgbt_tiny.md
if [ "${ABLATIONS:-0}" = "1" ]; then
  for cfg in configs/rgbt_tiny/ablations/*.yaml; do
    launch_train "$cfg"
    test_ckpts "$cfg"
  done
  for t in 4 5 6; do
    runs=(--run dyfclt=outputs/rgbt_tiny_dyfclt/test_metrics.json)
    for f in outputs/rgbt_tiny_t*/test_metrics.json; do n=$(basename "$(dirname "$f")"); runs+=(--run "${n#rgbt_tiny_}=$f"); done
    python tools/compare_with_paper.py --dataset rgbt_tiny --table ablation_table$t "${runs[@]}" --out outputs/compare_rgbt_tiny_table$t.md
  done
fi
