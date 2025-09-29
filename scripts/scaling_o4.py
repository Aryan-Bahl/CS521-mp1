#!/usr/bin/env python3
import subprocess
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-o4"
SRC = ROOT
EXEC = BUILD / "mp1_cpu"
PLOTS_DIR = ROOT / "plots"


def run(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return res.stdout

def parse_time_o3_as_o4(output):
    pattern = re.compile(r"Time taken for GEMM \(CPU,gemm_cpu_o3\): ([0-9]+\.?[0-9]*)ms")
    for line in output.splitlines():
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    raise RuntimeError("Timing for gemm_cpu_o3 not found in output")


def main():
    sizes = [100, 1000, 5000, 10000]
    times = []
    for n in sizes:
        print(f"Running o4 at N=M=K={n}...")
        out = run([str(EXEC), str(n), str(n), str(n)])
        print(out)
        t = parse_time_o3_as_o4(out)
        t_log = np.log(t)
        times.append(t_log)

    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, times, marker='o')
    ax.set_xlabel("Matrix size (N=M=K)")
    ax.set_ylabel("Runtime (log(ms))")
    ax.set_title("Scalability of o4 (flags-enabled o3)")
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    out_path = PLOTS_DIR / "scaling_o4.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    sys.exit(main())


