"""One-screen summary of a training run (send this output when asking for help).

python tools/summarize_run.py outputs/droneperson_dyfclt [--metrics AP50 AP50_tiny AP50_small]
"""
import argparse
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--metrics", nargs="*", default=None)
    args = ap.parse_args()
    mpath = os.path.join(args.run_dir, "metrics.jsonl")
    rows = [json.loads(l) for l in open(mpath)] if os.path.isfile(mpath) else []
    if not rows:
        print("no metrics.jsonl yet")
    keys = args.metrics
    if keys is None and rows:
        ev = next((r["eval"] for r in rows if r["eval"]), {})
        keys = [k for k in ("AP", "AP50", "AP_tiny", "AP50_tiny", "AP50_small", "AR") if k in ev]
    print(f"{'epoch':>5} {'loss':>8} " + " ".join(f"{k:>10}" for k in keys or []))
    for r in rows:
        ev = r.get("eval") or {}
        print(f"{r['epoch']:>5} {r['train'].get('loss', float('nan')):8.3f} " +
              " ".join(f"{ev.get(k, float('nan')):10.2f}" for k in keys or []))
    log = os.path.join(args.run_dir, "log.txt")
    if os.path.isfile(log):
        lines = open(log).read().splitlines()
        bands = [l for l in lines if "band boundaries" in l]
        if bands:
            print("last band boundaries:", bands[-1].split("band boundaries", 1)[1].strip())
        speed = [float(m.group(1)) for l in lines for m in [re.search(r"([\d.]+)s/it", l)] if m]
        if speed:
            print(f"speed: {speed[-1]:.2f} s/it (last), header: {lines[0]}")
        errs = [l for l in lines if "non-finite" in l or "Error" in l]
        for l in errs[-5:]:
            print("!!", l)
        best = [l for l in lines if l.startswith("done. best")]
        if best:
            print(best[-1])


if __name__ == "__main__":
    main()
