#!/usr/bin/env python3
"""
Tempest ablation runner (Python harness)

Runs 4 variants × N repetitions, storing:
- stdout.txt
- nsys.qdrep

Then parses stdout summaries and prints mean ± std per variant.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Parsing helpers
# -----------------------------
SUMMARY_PATTERNS = {
    "total_ingestion_time_sec": re.compile(r"^Total ingestion time:\s*([0-9eE+\-\.]+)\s*sec\s*$"),
    "mean_ingestion_time_sec": re.compile(r"^Mean ingestion time:\s*([0-9eE+\-\.]+)\s*sec\s*$"),
    "total_walk_time_sec": re.compile(r"^Total walk time:\s*([0-9eE+\-\.]+)\s*sec\s*$"),
    "mean_walk_time_per_batch_sec": re.compile(r"^Mean walk time/batch:\s*([0-9eE+\-\.]+)\s*sec\s*$"),
    "total_walks": re.compile(r"^Total walks:\s*([0-9]+)\s*$"),
    "final_avg_walk_length": re.compile(r"^Final avg walk length:\s*([0-9eE+\-\.]+)\s*$"),
    "throughput_walks_per_sec": re.compile(r"^Throughput:\s*([0-9eE+\-\.]+)\s*walks/sec\s*$"),
}


def parse_stdout_summary(stdout_path: Path) -> Dict[str, float]:
    text = stdout_path.read_text(errors="replace").splitlines()
    try:
        idx = next(i for i, line in enumerate(text) if line.strip() == "=== Summary ===")
    except StopIteration:
        raise ValueError(f"Could not find '=== Summary ===' in {stdout_path}")

    metrics: Dict[str, float] = {}
    for line in text[idx + 1:]:
        line = line.strip()
        if not line:
            continue
        for key, pat in SUMMARY_PATTERNS.items():
            m = pat.match(line)
            if m:
                metrics[key] = float(int(m.group(1))) if key == "total_walks" else float(m.group(1))
                break

    missing = [k for k in SUMMARY_PATTERNS if k not in metrics]
    if missing:
        raise ValueError(f"Missing fields {missing} while parsing {stdout_path}")

    return metrics


@dataclass(frozen=True)
class Variant:
    key: str
    description: str
    binary_rel: str
    picker: str
    kernel_mode: str

    def command(self, code_path: Path, dataset_csv: Path) -> List[str]:
        bin_path = (code_path / self.binary_rel).resolve()
        return [
            str(bin_path),
            str(dataset_csv),
            "1",
            self.picker,
            self.kernel_mode,
            "0",
            "5",
            "10",
            "3",
        ]


def which_or_die(prog: str) -> str:
    p = shutil.which(prog)
    if not p:
        raise FileNotFoundError(f"Could not find '{prog}' in PATH")
    return p


def tool_path(cuda_tools_path: Optional[str], name: str) -> str:
    if cuda_tools_path is None:
        return which_or_die(name)
    candidate = (Path(cuda_tools_path) / name).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find '{name}' at {candidate}")
    return str(candidate)


def run_cmd(cmd, cwd, stdout_path, stderr_path, dry_run):
    print("  $ " + " ".join(cmd))
    if dry_run:
        return 0
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=out,
            stderr=err,
            check=False,
            env=os.environ.copy(),
        )
    return proc.returncode


def format_mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-tools-path", default=None)
    ap.add_argument("--dataset-base-path", required=True)
    ap.add_argument("--code-path", default=str(Path.home()))
    ap.add_argument("--results-path", default=str(Path.cwd() / "ablation-results"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset_base = Path(args.dataset_base_path).expanduser().resolve()
    code_path = Path(args.code_path).expanduser().resolve()
    results_base = Path(args.results_path).expanduser().resolve()

    dataset_csv = dataset_base / "ml_tgbl-coin.csv"
    if not dataset_csv.exists():
        print(f"ERROR: dataset not found: {dataset_csv}", file=sys.stderr)
        return 2

    nsys = tool_path(args.cuda_tools_path, "nsys")

    variants = [
        Variant("v0_fullwalk_index_inkernel", "in-kernel RNG + full_walk + index (exponential_index)",
                "tempest/build/bin/ablation_streaming", "exponential_index", "full_walk"),
        Variant("v1_fullwalk_index_pregen", "pregenerated RNG + full_walk + index (exponential_index)",
                "tempest-pregenerated-rng/build/bin/ablation_streaming", "exponential_index", "full_walk"),
        Variant("v2_step_index_inkernel", "in-kernel RNG + step_based + index (exponential_index)",
                "tempest/build/bin/ablation_streaming", "exponential_index", "step_based"),
        Variant("v3_fullwalk_weight_inkernel", "in-kernel RNG + full_walk + weight (exponential_weight)",
                "tempest/build/bin/ablation_streaming", "exponential_weight", "full_walk"),
    ]

    metrics_by_variant = {v.key: {k: [] for k in SUMMARY_PATTERNS} for v in variants}

    print("========================================")
    print(" Tempest Ablation Runner")
    print("========================================")
    print(f"Dataset: {dataset_csv}")
    print(f"Code path: {code_path}")
    print(f"Results: {results_base}")
    print(f"Runs per variant: {args.runs}")
    print(f"nsys: {nsys}")
    print("")

    print("\n========================================")
    print(" GPU WARMUP RUN (baseline configuration)")
    print("========================================")

    warmup_cmd = variants[0].command(code_path, dataset_csv)
    print("Warmup variant:", variants[0].key)
    print("Warmup command:")
    print("  ", " ".join(warmup_cmd))

    if not args.dry_run:
        subprocess.run(warmup_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Warmup completed. Starting measured runs.\n")

    for run_id in range(1, args.runs + 1):
        run_dir = results_base / f"run-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"==============================")
        print(f"Run {run_id}/{args.runs} -> {run_dir}")
        print(f"==============================")

        for v in variants:
            var_dir = run_dir / v.key
            var_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nVariant: {v.key}")
            print(f"  {v.description}")
            print(f"  Output: {var_dir}")

            cmd = v.command(code_path, dataset_csv)

            rc = run_cmd(cmd, None, var_dir / "stdout.txt", var_dir / "stderr.txt", args.dry_run)
            if rc != 0:
                print(f"ERROR: normal run failed (rc={rc}) for {v.key} run-{run_id}", file=sys.stderr)
                return 3

            nsys_cmd = [
                nsys, "profile",
                "-o", str(var_dir / "nsys"),
                "--trace=cuda,nvtx,osrt",
                "--cuda-memory-usage=true",
                *cmd,
            ]
            rc = run_cmd(nsys_cmd, None, var_dir / "nsys_stdout.txt", var_dir / "nsys_stderr.txt", args.dry_run)
            if rc != 0:
                print(f"ERROR: nsys failed (rc={rc}) for {v.key} run-{run_id}", file=sys.stderr)
                return 4

            parsed = parse_stdout_summary(var_dir / "stdout.txt")
            for k, vval in parsed.items():
                metrics_by_variant[v.key][k].append(vval)

    print("\n========================================")
    print(" ABLATION SUMMARY (mean ± std)")
    print(f" Over {args.runs} runs per variant")
    print("========================================")

    for v in variants:
        m = metrics_by_variant[v.key]
        ingest_mean, ingest_std = format_mean_std(m["total_ingestion_time_sec"])
        walk_mean, walk_std = format_mean_std(m["total_walk_time_sec"])
        thr_mean, thr_std = format_mean_std(m["throughput_walks_per_sec"])
        print(f"\n{v.key}")
        print(f"  {v.description}")
        print(f"  Total ingestion time (sec): {ingest_mean:.3f} ± {ingest_std:.3f}")
        print(f"  Total walk time (sec):      {walk_mean:.3f} ± {walk_std:.3f}")
        print(f"  Throughput (walks/sec):     {thr_mean:.3e} ± {thr_std:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
