"""Compare reproduced metrics with the numbers reported in the paper (informational only).

Main table:
  python tools/compare_with_paper.py --dataset droneperson \
      --run dyfclt=outputs/droneperson_dyfclt/test_metrics.json \
      --run baseline=outputs/droneperson_baseline/test_metrics.json
Ablations (run names must match the keys in paper_results.yaml):
  python tools/compare_with_paper.py --dataset rgbt_tiny --table ablation_table4 \
      --run t4_none=outputs/rgbt_tiny_t4_none/test_metrics.json --run dyfclt=...
"""
import argparse
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_metrics(path):
    d = json.load(open(path))
    if "metrics" in d:
        return d["metrics"]
    return d


def fmt(v):
    return "-" if v is None else (f"{v:.1f}" if isinstance(v, (int, float)) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["rgbt_tiny", "droneperson", "flir"])
    ap.add_argument("--table", default=None, help="ablation_table4 | ablation_table5 | ablation_table6")
    ap.add_argument("--run", action="append", default=[], help="name=path/to/test_metrics.json")
    ap.add_argument("--out", help="write the markdown table here")
    args = ap.parse_args()
    paper = yaml.safe_load(open(os.path.join(ROOT, "paper_results.yaml")))[args.dataset]
    runs = {}
    for r in args.run:
        name, _, path = r.partition("=")
        runs[name] = load_metrics(path)

    lines = []
    if args.table:
        ref = paper[args.table]
        metrics = list(next(iter(ref.values())).keys())
        lines.append("| setting | " + " | ".join(f"{m} paper | {m} ours | Δ" for m in metrics) + " |")
        lines.append("|---" * (1 + 3 * len(metrics)) + "|")
        for name, pv in ref.items():
            ours = runs.get(name, {})
            cells = []
            for m in metrics:
                o = ours.get(m)
                cells += [fmt(pv[m]), fmt(o), fmt(o - pv[m]) if o is not None else "-"]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    else:
        ref = paper["dyfclt"]
        metrics = list(ref.keys())
        names = list(runs.keys())
        lines.append("| metric | paper DyFCLT | " + " | ".join(f"ours: {n} | Δ({n} - paper)" for n in names) + " |")
        lines.append("|---" * (2 + 2 * len(names)) + "|")
        for m in metrics + (["params_M"] if "params_M" in paper else []):
            pv = paper["dyfclt"].get(m) if m != "params_M" else paper["params_M"]
            cells = []
            for n in names:
                o = runs[n].get(m)
                cells += [fmt(o), fmt(o - pv) if (o is not None and pv is not None) else "-"]
            lines.append(f"| {m} | {fmt(pv)} | " + " | ".join(cells) + " |")
        if "best_prior_rgbt" in paper:
            b = paper["best_prior_rgbt"]
            lines.append("")
            lines.append(f"Best prior RGBT method in the paper: {b['method']} "
                         + ", ".join(f"{k}={v}" for k, v in b.items() if k != "method"))
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
