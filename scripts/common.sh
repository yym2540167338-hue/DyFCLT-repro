# Shared helpers. NGPU=<n> selects torchrun; extra overrides go in OPTS="key=value ...".
NGPU=${NGPU:-1}
PRETRAINED=${PRETRAINED:-pretrained/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth}
OPTS=${OPTS:-}
launch_train() {  # $1 = config
  local cfg=$1
  if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" tools/train.py -c "$cfg" --opts model.backbone.pretrained_path="$PRETRAINED" $OPTS
  else
    python tools/train.py -c "$cfg" --opts model.backbone.pretrained_path="$PRETRAINED" $OPTS
  fi
}
out_dir() { python - "$1" <<'PY'
import sys; sys.path.insert(0, '.')
from dyfclt.config import load_config
print(load_config(sys.argv[1])['output_dir'])
PY
}
test_ckpts() {  # evaluate last.pth and best.pth of a config
  local cfg=$1 d; d=$(out_dir "$cfg")
  python tools/test.py -c "$cfg" --ckpt "$d/last.pth" --out "$d/test_metrics_last.json"
  python tools/test.py -c "$cfg" --ckpt "$d/best.pth" --out "$d/test_metrics.json"
}
