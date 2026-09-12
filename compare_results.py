#!/usr/bin/env python3
"""
Parse BatteryLife run logs, compute mean +/- std across seeds, and compare
against the published BatteryLife paper benchmark table (README.md).

Usage:
    python3 compare_results.py results_logs/CPMLP_CALB*.log --model CPMLP --dataset_family CALB
"""
import argparse, glob, re, statistics, sys

PUBLISHED = {
    ("CPMLP", "Li-ion"):        {"mape": (0.179, 0.003), "acc15": (0.620, 0.004)},
    ("CPMLP", "Zn-ion"):        {"mape": (0.558, 0.034), "acc15": (0.297, 0.084)},
    ("CPMLP", "Na-ion"):        {"mape": (0.274, 0.026), "acc15": (0.337, 0.038)},
    ("CPMLP", "CALB"):          {"mape": (0.140, 0.009), "acc15": (0.704, 0.053)},
    ("CPTransformer", "Li-ion"): {"mape": (0.184, 0.003), "acc15": (0.573, 0.016)},
    ("CPTransformer", "Zn-ion"): {"mape": (0.515, 0.067), "acc15": (0.202, 0.084)},
    ("CPTransformer", "Na-ion"): {"mape": (0.255, 0.036), "acc15": (0.406, 0.084)},
    ("CPTransformer", "CALB"):   {"mape": (0.149, 0.005), "acc15": (0.672, 0.107)},
}

LINE_RE = re.compile(
    r"Best model performance:\s*Test MAE:\s*([\d.]+)\s*\|\s*Test RMSE:\s*([\d.]+)\s*\|\s*"
    r"Test MAPE:\s*([\d.]+)\s*\|\s*Test 15%-accuracy:\s*([\d.]+)\s*\|\s*Test 10%-accuracy:\s*([\d.]+)"
)

def parse_log(path):
    with open(path) as f:
        text = f.read()
    matches = LINE_RE.findall(text)
    if not matches:
        print(f"  WARNING: no 'Best model performance' line found in {path}", file=sys.stderr)
        return None
    mae, rmse, mape, acc15, acc10 = matches[0]
    return {"path": path, "mae": float(mae), "rmse": float(rmse), "mape": float(mape),
            "acc15_pct": float(acc15), "acc10_pct": float(acc10)}

def fmt_mean_std(values):
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset_family", required=True)
    args = ap.parse_args()

    paths = []
    for pattern in args.logs:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("No log files matched.", file=sys.stderr); sys.exit(1)

    results = []
    print(f"Parsing {len(paths)} log file(s):")
    for p in paths:
        r = parse_log(p)
        if r is None:
            continue
        results.append(r)
        print(f"  {p}: Test MAPE={r['mape']:.4f}  Test 15%-acc={r['acc15_pct']:.2f}%  "
              f"Test MAE={r['mae']:.4f}  Test RMSE={r['rmse']:.4f}")

    if not results:
        print("No parseable results found.", file=sys.stderr); sys.exit(1)

    mape_mean, mape_std = fmt_mean_std([r["mape"] for r in results])
    acc15_mean, acc15_std = fmt_mean_std([r["acc15_pct"] / 100.0 for r in results])

    print(f"\n=== Your results across {len(results)} run(s) ===")
    print(f"  Test MAPE:      {mape_mean:.3f} +/- {mape_std:.3f}")
    print(f"  Test 15%-Acc:   {acc15_mean:.3f} +/- {acc15_std:.3f}")

    key = (args.model, args.dataset_family)
    pub = PUBLISHED.get(key)
    if pub is None:
        print(f"\n(No published row on file for {key} -- add it to PUBLISHED if you have it.)")
        return

    pub_mape_mean, pub_mape_std = pub["mape"]
    pub_acc_mean, pub_acc_std = pub["acc15"]
    print(f"\n=== Published BatteryLife paper ({args.model} / {args.dataset_family}) ===")
    print(f"  Test MAPE:      {pub_mape_mean:.3f} +/- {pub_mape_std:.3f}")
    print(f"  Test 15%-Acc:   {pub_acc_mean:.3f} +/- {pub_acc_std:.3f}")

    print(f"\n=== Delta (yours - published) ===")
    print(f"  Test MAPE:      {mape_mean - pub_mape_mean:+.3f}")
    print(f"  Test 15%-Acc:   {acc15_mean - pub_acc_mean:+.3f}")

if __name__ == "__main__":
    main()
