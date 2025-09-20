#!/usr/bin/env python3
import subprocess
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-ablation"
SRC = ROOT
EXEC = BUILD / "mp1_cpu"
PLOTS_DIR = ROOT / "plots"


def run(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return res.stdout


def parse_times(output):
    times = {}
    pattern = re.compile(r"Time taken for GEMM \(CPU,(gemm_cpu_o\d)\): ([0-9]+\.?[0-9]*)ms")
    for line in output.splitlines():
        m = pattern.search(line)
        if m:
            times[m.group(1)] = float(m.group(2))
    return times


def main():
    sizes = [100, 1000]

    results = {}
    for n in sizes:
        print(f"Running N=M=K={n}...")
        out = run([str(EXEC), str(n), str(n), str(n)])
        print(out)
        times = parse_times(out)
        for key in ["gemm_cpu_o0", "gemm_cpu_o1", "gemm_cpu_o2", "gemm_cpu_o3"]:
            if key not in times:
                raise RuntimeError(f"Missing timing for {key} at size {n}")
        results[n] = times

    labels = ["o1", "o2", "o3"]
    x = np.arange(len(labels))
    width = 0.35

    speedups_100 = [results[100]["gemm_cpu_o0"] / results[100][f"gemm_cpu_{lbl}"] for lbl in labels]
    speedups_1000 = [results[1000]["gemm_cpu_o0"] / results[1000][f"gemm_cpu_{lbl}"] for lbl in labels]

    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, speedups_100, width, label="N=M=K=100")
    ax.bar(x + width/2, speedups_1000, width, label="N=M=K=1000")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup vs o0 (×)")
    ax.set_title("Ablation: Speedup by Optimization and Size")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    out_path = PLOTS_DIR / "ablation_speedup.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    sys.exit(main())


