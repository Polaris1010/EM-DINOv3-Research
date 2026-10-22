"""Experiment Controller for SARR (Source-Adaptive Representation Routing).

Orchestrates the execution of SARR and ablation runs on locked MV-Generalist-v2 data:
  1. sarr (3 decoder seeds: 20261014, 20261015, 20261016)
  2. uniform_avg (1 seed: 20261014) - Fixed 50/50 blend baseline
  3. global_gate (1 seed: 20261014) - Single learned scalar alpha baseline
  (Optional) p0_only (1 seed) - G-P0 reproduction
  (Optional) b100_only (1 seed) - G-B100 reproduction

Enforces strict GPU access policy:
  - GPU 0 is strictly FORBIDDEN. Only GPUs 1, 2, or 3 are permitted.
  - Maps CUDA_VISIBLE_DEVICES to permitted physical GPU.

Collects metrics, computes source-equal-weight Macro Dice, generates:
  - sarr_summary.json
  - sarr_comparison.csv
  - sarr_comparison.md
"""

from pathlib import Path
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

# Path resolution
R = Path(os.environ.get('SARR_ROOT', Path(__file__).resolve().parents[2]))
SRC_DIR = R / 'src'
SARR_DIR = SRC_DIR / 'sarr'
TRAIN_SCRIPT = SARR_DIR / 'train_sarr.py'

# Server Python interpreter path with local fallback
SERVER_PYTHON = Path('/opt/data/3_117/tanghr/envs/meddinov3/bin/python')
PYTHON_BIN = SERVER_PYTHON if SERVER_PYTHON.exists() else Path(sys.executable)

# Core ablation runs (5 runs)
DEFAULT_ABLATION_RUNS = [
    {'mode': 'sarr', 'seed': 20261014, 'description': 'Full Spatial Router (Seed 1)'},
    {'mode': 'sarr', 'seed': 20261015, 'description': 'Full Spatial Router (Seed 2)'},
    {'mode': 'sarr', 'seed': 20261016, 'description': 'Full Spatial Router (Seed 3)'},
    {'mode': 'uniform_avg', 'seed': 20261014, 'description': 'Fixed alpha=0.5 Blend'},
    {'mode': 'global_gate', 'seed': 20261014, 'description': 'Single Learned Scalar Gate'},
]

BASELINE_RUNS = [
    {'mode': 'p0_only', 'seed': 20261014, 'description': 'G-P0 Single Encoder Baseline'},
    {'mode': 'b100_only', 'seed': 20261014, 'description': 'G-B100 Single Encoder Baseline'},
]


def atomic(p: Path, obj: dict):
    """Write JSON file atomically."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f'.tmp_{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)


def validate_gpu_assignment(gpu: int) -> int:
    """Validate that requested GPU index is allowed (1, 2, or 3) and never 0."""
    if gpu == 0:
        raise ValueError("CRITICAL: GPU 0 is strictly forbidden by research policy! Must use GPU 1, 2, or 3.")
    if gpu not in (1, 2, 3):
        raise ValueError(f"Requested GPU {gpu} is outside the allowed set [1, 2, 3].")
    return gpu


def resolve_physical_gpu_env(gpu: int) -> dict:
    """Map physical GPU index to CUDA_VISIBLE_DEVICES environment variable."""
    validate_gpu_assignment(gpu)
    env = dict(os.environ)

    # If nvidia-smi is available, retrieve the specific GPU UUID for robustness
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
            text=True
        )
        gpu_map = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2:
                gpu_map[int(parts[0])] = parts[1]

        if gpu in gpu_map:
            env['CUDA_VISIBLE_DEVICES'] = gpu_map[gpu]
        else:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    except Exception:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Thread and cache safety
    env['OMP_NUM_THREADS'] = '2'
    env['MKL_NUM_THREADS'] = '2'
    env['PYTHONUNBUFFERED'] = '1'
    return env


def execute_run(
    mode: str,
    seed: int,
    steps: int,
    out_dir: Path,
    gpu: int,
    smoke: bool = False,
) -> bool:
    """Launch a single training or smoke run as a subprocess."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / ('smoke.log' if smoke else 'train.log')
    env = resolve_physical_gpu_env(gpu)

    cmd = [
        str(PYTHON_BIN),
        str(TRAIN_SCRIPT),
        '--mode', mode,
        '--seed', str(seed),
        '--steps', str(steps),
        '--out', str(out_dir),
    ]
    if smoke:
        cmd.append('--smoke')

    print(f"\n{'='*70}")
    print(f"Launching Run: mode={mode} seed={seed} steps={steps} on GPU {gpu}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    print(f"{'='*70}\n", flush=True)

    t0 = time.time()
    with log_path.open('w', encoding='utf-8') as log_f:
        p = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ret = p.wait()

    elapsed = time.time() - t0
    if ret != 0:
        print(f"FAILED: mode={mode} seed={seed} (exit code {ret}) after {elapsed:.1f}s. Check {log_path}", flush=True)
        return False

    print(f"COMPLETED: mode={mode} seed={seed} in {elapsed:.1f}s", flush=True)
    return True


def collect_run_metrics(run_dir: Path, smoke: bool = False) -> dict:
    """Extract metrics from run completion files."""
    if smoke:
        smoke_file = run_dir / 'smoke.json'
        if smoke_file.exists():
            return json.loads(smoke_file.read_text(encoding='utf-8'))
        return {'status': 'MISSING'}

    metrics_file = run_dir / 'metrics.json'
    if not metrics_file.exists():
        # Check if progress.json exists for partial info
        progress_file = run_dir / 'progress.json'
        if progress_file.exists():
            prog = json.loads(progress_file.read_text(encoding='utf-8'))
            return {'status': 'INCOMPLETE', 'last_progress': prog}
        return {'status': 'MISSING'}

    data = json.loads(metrics_file.read_text(encoding='utf-8'))
    val = data.get('final_validation', {})

    return {
        'status': 'COMPLETE',
        'mode': data.get('mode'),
        'seed': data.get('seed'),
        'steps': data.get('completed_steps', data.get('steps')),
        'seconds': data.get('seconds'),
        'trainable_parameters': data.get('trainable_parameters'),
        'source_macro_dice': val.get('source_macro_dice', val.get('descriptive_all_source_macro_dice')),
        'source_means': val.get('source_means', {}),
        'routing_stats': val.get('routing_stats', {}),
        'overall_mean_alpha': val.get('overall_mean_alpha'),
    }


def compile_experiment_summary(
    runs: list,
    base_out: Path,
    smoke: bool = False,
) -> dict:
    """Aggregate results from all runs into summary dictionary and tables."""
    results = []
    by_mode = collections.defaultdict(list)

    for run_spec in runs:
        mode = run_spec['mode']
        seed = run_spec['seed']
        run_dir = base_out / mode / f"seed{seed}"
        info = collect_run_metrics(run_dir, smoke=smoke)
        info['mode'] = mode
        info['seed'] = seed
        info['description'] = run_spec.get('description', '')
        info['run_dir'] = str(run_dir)
        results.append(info)
        if info.get('status') == 'COMPLETE':
            by_mode[mode].append(info)

    # Compute aggregate stats for modes with multiple seeds (e.g. sarr)
    mode_summaries = {}
    for mode, mode_runs in by_mode.items():
        dices = [r['source_macro_dice'] for r in mode_runs if r.get('source_macro_dice') is not None]
        times = [r['seconds'] for r in mode_runs if r.get('seconds') is not None]

        # Aggregate per-source Dice
        source_keys = sorted(mode_runs[0].get('source_means', {}).keys()) if mode_runs else []
        src_agg = {}
        for src in source_keys:
            src_vals = [r['source_means'][src] for r in mode_runs if src in r.get('source_means', {})]
            src_agg[src] = {
                'mean': float(np.mean(src_vals)) if src_vals else None,
                'std': float(np.std(src_vals, ddof=1)) if len(src_vals) > 1 else 0.0,
            }

        # Aggregate routing alpha per source
        routing_agg = {}
        for src in source_keys:
            alphas = [
                r['routing_stats'][src]['mean_alpha']
                for r in mode_runs
                if src in r.get('routing_stats', {}) and 'mean_alpha' in r['routing_stats'][src]
            ]
            if alphas:
                routing_agg[src] = {
                    'mean_alpha': float(np.mean(alphas)),
                    'std_alpha': float(np.std(alphas, ddof=1)) if len(alphas) > 1 else 0.0,
                }

        mode_summaries[mode] = {
            'n_runs': len(mode_runs),
            'macro_dice_mean': float(np.mean(dices)) if dices else None,
            'macro_dice_std': float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0,
            'macro_dice_values': dices,
            'mean_time_seconds': float(np.mean(times)) if times else None,
            'trainable_parameters': mode_runs[0].get('trainable_parameters') if mode_runs else None,
            'per_source': src_agg,
            'routing_alpha': routing_agg,
        }

    # Reference numbers from official MV-3 benchmark for comparison
    reference_baselines = {
        'G-P0': {'macro_dice': 0.813043, 'source': 'MV-3 official 38-dev macro Dice'},
        'G-B100': {'macro_dice': 0.827307, 'source': 'MV-3 official 38-dev macro Dice'},
    }

    summary = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'experiment': 'SARR (Source-Adaptive Representation Routing)',
        'runs': results,
        'mode_summaries': mode_summaries,
        'reference_baselines': reference_baselines,
    }

    return summary


def write_comparison_tables(summary: dict, base_out: Path):
    """Generate Markdown and CSV comparison tables."""
    atomic(base_out / 'sarr_summary.json', summary)

    mode_sums = summary.get('mode_summaries', {})
    sarr_mean = mode_sums.get('sarr', {}).get('macro_dice_mean')

    # Prepare Markdown comparison
    md_lines = [
        "# SARR (Source-Adaptive Representation Routing) Experiment Results",
        "",
        "## Overall Model Comparison (12 EM Sources, 38 Dev Regions)",
        "",
        "| Architecture / Mode | Trainable Params | Seeds | Macro Dice (Mean ± SD) | Delta vs SARR | Delta vs G-B100 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    # Baseline rows
    b100_ref = summary['reference_baselines']['G-B100']['macro_dice']
    p0_ref = summary['reference_baselines']['G-P0']['macro_dice']

    rows = []
    # P0 baseline
    p0_dice = mode_sums.get('p0_only', {}).get('macro_dice_mean', p0_ref)
    p0_d_sarr = f"{p0_dice - sarr_mean:+.4f}" if sarr_mean else "N/A"
    p0_d_b100 = f"{p0_dice - b100_ref:+.4f}"
    rows.append(("P0 Only (Natural DINOv3)", "44,576,514", "1 (or ref)", f"{p0_dice:.4f}", p0_d_sarr, p0_d_b100))

    # B100 baseline
    b100_dice = mode_sums.get('b100_only', {}).get('macro_dice_mean', b100_ref)
    b100_d_sarr = f"{b100_dice - sarr_mean:+.4f}" if sarr_mean else "N/A"
    rows.append(("B100k Only (EM DINOv3)", "44,576,514", "1 (or ref)", f"{b100_dice:.4f}", b100_d_sarr, "0.0000"))

    # Uniform Avg
    if 'uniform_avg' in mode_sums:
        u_dice = mode_sums['uniform_avg']['macro_dice_mean']
        u_d_sarr = f"{u_dice - sarr_mean:+.4f}" if sarr_mean and u_dice else "N/A"
        u_d_b100 = f"{u_dice - b100_ref:+.4f}" if u_dice else "N/A"
        dice_str = f"{u_dice:.4f}" if u_dice else "Pending"
        rows.append(("Uniform Average (Fixed alpha=0.5)", "44,576,514", "1", dice_str, u_d_sarr, u_d_b100))

    # Global Gate
    if 'global_gate' in mode_sums:
        g_dice = mode_sums['global_gate']['macro_dice_mean']
        g_d_sarr = f"{g_dice - sarr_mean:+.4f}" if sarr_mean and g_dice else "N/A"
        g_d_b100 = f"{g_dice - b100_ref:+.4f}" if g_dice else "N/A"
        dice_str = f"{g_dice:.4f}" if g_dice else "Pending"
        rows.append(("Global Gate (Scalar alpha)", "44,576,515", "1", dice_str, g_d_sarr, g_d_b100))

    # SARR
    if 'sarr' in mode_sums:
        s_mean = mode_sums['sarr']['macro_dice_mean']
        s_std = mode_sums['sarr']['macro_dice_std']
        s_n = mode_sums['sarr']['n_runs']
        s_d_b100 = f"{s_mean - b100_ref:+.4f}" if s_mean else "N/A"
        dice_str = f"**{s_mean:.4f} ± {s_std:.4f}**" if s_mean else "Pending"
        rows.append(("**SARR (Spatial Router)**", "46,674,435", str(s_n), dice_str, "**Baseline**", f"**{s_d_b100}**"))

    for r in rows:
        md_lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")

    # Routing weight section
    if 'sarr' in mode_sums and mode_sums['sarr'].get('routing_alpha'):
        md_lines.extend([
            "",
            "## SARR Routing Weight Statistics per EM Source",
            "",
            "Routing weight $\\alpha \\in (0, 1)$ indicates encoder preference: $\\alpha \\to 1$ selects P0 (natural), $\\alpha \\to 0$ selects B100k (MitoVerse EM).",
            "",
            "| Source ID | Mean Routing Weight $\\alpha$ | Std $\\alpha$ | SARR Preferred Encoder |",
            "| :--- | :---: | :---: | :--- |",
        ])
        for src, stat in sorted(mode_sums['sarr']['routing_alpha'].items()):
            alpha_m = stat.get('mean_alpha', 0.5)
            alpha_s = stat.get('std_alpha', 0.0)
            pref = "P0 (Natural)" if alpha_m > 0.55 else ("B100k (EM Continual)" if alpha_m < 0.45 else "Balanced Hybrid")
            md_lines.append(f"| `{src}` | {alpha_m:.4f} | {alpha_s:.4f} | {pref} |")

    md_content = '\n'.join(md_lines) + '\n'
    (base_out / 'sarr_comparison.md').write_text(md_content, encoding='utf-8')

    # Prepare CSV export
    csv_path = base_out / 'sarr_comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Mode', 'Trainable_Params', 'Seeds', 'Macro_Dice_Mean', 'Macro_Dice_Std', 'Delta_vs_SARR', 'Delta_vs_B100'])
        for r in rows:
            clean_mode = r[0].replace('**', '')
            clean_dice = r[3].replace('**', '')
            parts = clean_dice.split(' ± ')
            mean_val = parts[0] if parts else clean_dice
            std_val = parts[1] if len(parts) > 1 else '0.0000'
            writer.writerow([clean_mode, r[1], r[2], mean_val, std_val, r[4].replace('**', ''), r[5].replace('**', '')])

    print("\n" + md_content, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run SARR experiment suite")
    parser.add_argument('--gpu', type=int, default=1, choices=[1, 2, 3],
                        help="Physical GPU index to use (FORBIDDEN: 0. Choices: 1, 2, 3)")
    parser.add_argument('--steps', type=int, default=40000, help="Training steps per run (default 40000)")
    parser.add_argument('--out', type=Path, default=None, help="Base output directory")
    parser.add_argument('--smoke', action='store_true', help="Run smoke test for each configuration")
    parser.add_argument('--include-baselines', action='store_true',
                        help="Also train single-encoder p0_only and b100_only runs")
    parser.add_argument('--collect-only', action='store_true',
                        help="Skip execution and only collect existing results into summary table")
    args = parser.parse_args()

    # Enforce GPU 0 policy
    validate_gpu_assignment(args.gpu)

    if args.out is None:
        args.out = R / 'experiments/SARR'
    args.out.mkdir(parents=True, exist_ok=True)

    runs = list(DEFAULT_ABLATION_RUNS)
    if args.include_baselines:
        runs.extend(BASELINE_RUNS)

    print(f"SARR Controller started. Base output: {args.out}")
    print(f"Target GPU: {args.gpu} (Physical GPU 0 is strictly forbidden)")
    print(f"Total planned runs: {len(runs)}")

    if not args.collect_only:
        for idx, r in enumerate(runs):
            mode = r['mode']
            seed = r['seed']
            run_dir = args.out / mode / f"seed{seed}"

            # Check if run already completed
            target_file = run_dir / ('smoke.json' if args.smoke else 'metrics.json')
            if target_file.exists():
                print(f"[{idx+1}/{len(runs)}] Run {mode} seed={seed} already complete. Skipping.")
                continue

            success = execute_run(
                mode=mode,
                seed=seed,
                steps=args.steps,
                out_dir=run_dir,
                gpu=args.gpu,
                smoke=args.smoke,
            )
            if not success:
                print(f"Aborting experiment pipeline due to failure in {mode} seed={seed}.")
                break

    # Compile results and write comparison tables
    summary = compile_experiment_summary(runs, args.out, smoke=args.smoke)
    write_comparison_tables(summary, args.out)
    print("SARR experiment controller finished successfully.")


if __name__ == '__main__':
    main()
