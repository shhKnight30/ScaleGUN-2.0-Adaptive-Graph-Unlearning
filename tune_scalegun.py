"""
ScaleGUN Parameter Tuning Script
=================================
Runs grid search over all key parameters.
All output goes to tuning_master.log + tuning_results.csv

Usage:
    python tune_scalegun.py                    # quick mode
    python tune_scalegun.py --mode paper       # paper settings only
    python tune_scalegun.py --mode quick       # fast subset (~18 runs)
    python tune_scalegun.py --mode full        # all combinations (~200+ runs)
    python tune_scalegun.py --mode custom      # edit CUSTOM_GRID below
    python tune_scalegun.py --mode quick --dry-run   # preview combinations
"""

import subprocess
import itertools
import csv
import os
import re
import sys
import time
import logging
import argparse
from datetime import datetime

# ─── LOGGING SETUP ────────────────────────────────────────────────
LOG_FILE     = "./tuning_master.log"
RESULTS_FILE = "./tuning_results.csv"
RUNS_LOG_DIR = "./tuning_logs"
os.makedirs(RUNS_LOG_DIR, exist_ok=True)

logger = logging.getLogger("scalegun_tuner")
logger.setLevel(logging.DEBUG)

# File handler — ALL output (DEBUG + INFO + WARNING + ERROR)
fh = logging.FileHandler(LOG_FILE, mode="a")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

# Console handler — only INFO and above (minimal screen output)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

# ─── PARAMETER GRIDS ──────────────────────────────────────────────

PAPER_GRID = {
    "dataset"           : ["cora"],
    "rmax"              : ["1e-7"],
    "lam"               : ["1e-2"],
    "std"               : ["0.1"],
    "eps"               : ["1.0"],
    "prop_step"         : ["2"],
    "weight_mode"       : ["test"],
    "lr"                : ["1.0"],
    "optimizer"         : ["LBFGS"],
    "epochs"            : ["100"],
    "num_batch_removes" : ["2000"],
    "num_removes"       : ["1"],
    "seed"              : ["0"],
}

QUICK_GRID = {
    "dataset"           : ["cora"],
    "rmax"              : ["1e-7","1e-8" ,],
    "lam"               : ["1e-2", "5*(1e-3)"],
    "std"               : ["0.08","0.05"],
    "eps"               : ["1.0"],
    "prop_step"         : ["3"],
    "weight_mode"       : ["avg"],
    "lr"                : ["0.1"],
    "optimizer"         : ["Adam","LBFGS"],
    "epochs"            : ["200"],
    "num_batch_removes" : ["200"],
    "num_removes"       : ["1"],
    "seed"              : ["0"],
}

FULL_GRID = {
    "dataset"           : ["cora"],
    "rmax"              : ["1e-5", "1e-7", "1e-9", "1e-10"],
    "lam"               : ["1e-1", "1e-2", "1e-3", "1e-4"],
    "std"               : ["0", "0.05", "0.1", "0.5", "1.0"],
    "eps"               : ["0.5", "1.0", "2.0"],
    "prop_step"         : ["2", "3"],
    "weight_mode"       : ["test", "avg", "decay"],
    "lr"                : ["1.0", "0.1", "0.01"],
    "optimizer"         : ["LBFGS", "Adam"],
    "epochs"            : ["100", "300"],
    "num_batch_removes" : ["500"],
    "num_removes"       : ["1", "5"],
    "seed"              : ["0", "1", "2"],
}

# Edit this freely for your own experiments
CUSTOM_GRID = {
    "dataset"           : ["cora"],
    "rmax"              : ["1e-9", "1e-10"],
    "lam"               : ["1e-2", "1e-3"],
    "std"               : ["0", "0.1"],
    "eps"               : ["1.0"],
    "prop_step"         : ["2"],
    "weight_mode"       : ["avg"],
    "lr"                : ["0.01"],
    "optimizer"         : ["Adam"],
    "epochs"            : ["300"],
    "num_batch_removes" : ["500"],
    "num_removes"       : ["1"],
    "seed"              : ["0"],
}

# ─── PARSE SCALEGUN LOG ───────────────────────────────────────────
def parse_scalegun_log(log_path):
    """Extract metrics from ScaleGUN's log file."""
    metrics = {
        "initial_val_acc"  : None,
        "initial_test_acc" : None,
        "final_val_acc"    : None,
        "final_test_acc"   : None,
        "avg_update_cost"  : None,
        "avg_unlearn_cost" : None,
        "avg_tot_cost"     : None,
        "num_retrain"      : None,
        "train_cost"       : None,
        "total_time"       : None,
        "completed"        : False,
        "error"            : None,
    }
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        content = "".join(lines)

        for line in lines:
            if "Validation accuracy:" in line:
                m = re.search(r"Validation accuracy: ([0-9.]+)", line)
                if m: metrics["initial_val_acc"] = float(m.group(1))

            if "Test accuracy:" in line and "Iteration" not in line:
                m = re.search(r"Test accuracy: ([0-9.]+)", line)
                if m: metrics["initial_test_acc"] = float(m.group(1))

            if "first train cost:" in line:
                m = re.search(r"first train cost: ([0-9.]+)s", line)
                if m: metrics["train_cost"] = float(m.group(1))

            if "Iteration" in line and "Edge del" in line:
                vm = re.search(r"Val acc = ([0-9.]+)", line)
                tm = re.search(r"Test acc = ([0-9.]+)", line)
                rm = re.search(r"num_retrain: (\d+)", line)
                um = re.search(r"avg unlearn cost:([0-9.]+)", line)
                cm = re.search(r"avg tot cost:([0-9.]+)", line)
                uc = re.search(r"avg update cost: ([0-9.]+)", line)
                if vm: metrics["final_val_acc"]    = float(vm.group(1))
                if tm: metrics["final_test_acc"]   = float(tm.group(1))
                if rm: metrics["num_retrain"]      = int(rm.group(1))
                if um: metrics["avg_unlearn_cost"] = float(um.group(1))
                if cm: metrics["avg_tot_cost"]     = float(cm.group(1))
                if uc: metrics["avg_update_cost"]  = float(uc.group(1))

            if "tot cost:" in line and "first" not in line:
                m = re.search(r"tot cost: ([0-9.]+)s", line)
                if m: metrics["total_time"] = float(m.group(1))

        if "tot cost:" in content:
            metrics["completed"] = True

        if "Traceback" in content or "IndexError" in content:
            for line in lines:
                if "Error" in line or "Traceback" in line:
                    metrics["error"] = line.strip()[:120]
                    break

    except Exception as e:
        metrics["error"] = str(e)

    return metrics

# ─── RUN ONE EXPERIMENT ───────────────────────────────────────────
def run_experiment(params, run_id, total_runs):
    """Run a single ScaleGUN experiment and return metrics."""

    logger.info(f"{'─'*60}")
    logger.info(f"RUN {run_id}/{total_runs}")
    logger.info(f"  rmax={params['rmax']}  lam={params['lam']}  std={params['std']}  eps={params['eps']}")
    logger.info(f"  prop_step={params['prop_step']}  weight_mode={params['weight_mode']}")
    logger.info(f"  lr={params['lr']}  optimizer={params['optimizer']}  epochs={params['epochs']}")
    logger.info(f"  num_batch_removes={params['num_batch_removes']}  num_removes={params['num_removes']}  seed={params['seed']}")

    cmd = [
        "python", "edge_exp.py",
        "--dataset",           params["dataset"],
        "--rmax",              params["rmax"],
        "--lam",               params["lam"],
        "--std",               params["std"],
        "--eps",               params["eps"],
        "--prop_step",         params["prop_step"],
        "--weight_mode",       params["weight_mode"],
        "--lr",                params["lr"],
        "--optimizer",         params["optimizer"],
        "--epochs",            params["epochs"],
        "--num_batch_removes", params["num_batch_removes"],
        "--num_removes",       params["num_removes"],
        "--seed",              params["seed"],
        "--dev",               "-1",
        "--disp",              "100",
        "--edge_idx_start",    "0",
    ]

    logger.debug(f"  CMD: {' '.join(cmd)}")

    # Per-run log file captures raw subprocess output
    run_log = os.path.join(
        RUNS_LOG_DIR,
        f"run_{run_id:04d}"
        f"_rmax{params['rmax']}"
        f"_lam{params['lam']}"
        f"_std{params['std']}"
        f"_opt{params['optimizer']}"
        f"_seed{params['seed']}.log"
    )

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        wall_time = time.time() - start_time

        # Save raw subprocess output to per-run log
        with open(run_log, "w") as f:
            f.write(f"CMD: {' '.join(cmd)}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

        logger.debug(f"  Subprocess stdout+stderr saved to: {run_log}")
        logger.debug(f"  Return code: {result.returncode}  Wall time: {wall_time:.1f}s")

        if result.returncode != 0:
            logger.warning(f"  Non-zero return code: {result.returncode}")
            for line in result.stderr.splitlines()[-5:]:
                logger.warning(f"  STDERR: {line}")

    except subprocess.TimeoutExpired:
        wall_time = time.time() - start_time
        logger.error(f"  TIMEOUT after 600s")
        with open(run_log, "w") as f:
            f.write("TIMEOUT after 600 seconds\n")

    # Find the latest ScaleGUN analysis log (ScaleGUN creates its own)
    analysis_dir = f"./analysis/{params['dataset']}/edge/"
    scalegun_log = None
    if os.path.exists(analysis_dir):
        logs = sorted(
            [f for f in os.listdir(analysis_dir) if f.endswith(".log")],
            key=lambda x: os.path.getmtime(os.path.join(analysis_dir, x))
        )
        if logs:
            scalegun_log = os.path.join(analysis_dir, logs[-1])
            logger.debug(f"  Parsing ScaleGUN log: {scalegun_log}")

    # Parse metrics
    metrics = parse_scalegun_log(scalegun_log) if scalegun_log else parse_scalegun_log(run_log)
    metrics["wall_time"] = round(wall_time, 2)

    # Log result summary into master log
    if metrics["completed"]:
        logger.info(f"  ✅ COMPLETED")
        logger.info(f"     Initial test acc  = {metrics['initial_test_acc']}")
        logger.info(f"     Final   test acc  = {metrics['final_test_acc']}")
        logger.info(f"     Num retrains      = {metrics['num_retrain']}")
        logger.info(f"     Avg total cost    = {metrics['avg_tot_cost']}s/edge")
        logger.info(f"     Wall time         = {metrics['wall_time']}s")
    else:
        logger.warning(f"  ❌ DID NOT COMPLETE")
        logger.warning(f"     Error: {metrics.get('error', 'unknown')}")

    return metrics

# ─── SAVE CSV ─────────────────────────────────────────────────────
def save_csv(all_results):
    if not all_results:
        return
    fieldnames = list(all_results[0].keys())
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logger.debug(f"CSV updated: {RESULTS_FILE} ({len(all_results)} rows)")

# ─── LOG FINAL SUMMARY ────────────────────────────────────────────
def log_summary(all_results):
    completed = [r for r in all_results
                 if str(r.get("completed")).lower() == "true"
                 and r.get("final_test_acc") is not None]

    if not completed:
        logger.warning("No completed runs to summarize.")
        return

    top10 = sorted(completed, key=lambda x: x.get("final_test_acc", 0), reverse=True)[:10]

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY — TOP 10 BY FINAL TEST ACCURACY")
    logger.info("=" * 70)
    logger.info(
        f"  {'Rank':<5} {'FinalTest':>10} {'InitTest':>10} {'Retrains':>9} "
        f"{'TotCost':>9} {'rmax':<9} {'lam':<7} {'std':<6} {'lr':<7} {'opt':<7} {'seed'}"
    )
    logger.info(
        f"  {'─'*5} {'─'*10} {'─'*10} {'─'*9} {'─'*9} "
        f"{'─'*9} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*4}"
    )
    for i, r in enumerate(top10):
        logger.info(
            f"  {i+1:<5} "
            f"{r.get('final_test_acc', 0):>10.4f} "
            f"{r.get('initial_test_acc') or 0:>10.4f} "
            f"{str(r.get('num_retrain', '?')):>9} "
            f"{r.get('avg_tot_cost') or 0:>9.4f} "
            f"{str(r.get('rmax', '?')):<9} "
            f"{str(r.get('lam', '?')):<7} "
            f"{str(r.get('std', '?')):<6} "
            f"{str(r.get('lr', '?')):<7} "
            f"{str(r.get('optimizer', '?')):<7} "
            f"{r.get('seed', '?')}"
        )

    best = top10[0]
    logger.info("")
    logger.info("🏆 BEST CONFIGURATION:")
    for key in ["rmax", "lam", "std", "eps", "prop_step",
                "weight_mode", "lr", "optimizer", "epochs", "num_removes", "seed"]:
        logger.info(f"   {key:<20} = {best.get(key)}")
    logger.info("")
    logger.info(f"   Initial test acc : {best.get('initial_test_acc')}")
    logger.info(f"   Final   test acc : {best.get('final_test_acc')}")
    logger.info(f"   Num retrains     : {best.get('num_retrain')}")
    logger.info(f"   Avg total cost   : {best.get('avg_tot_cost')}s/edge")
    logger.info("")
    logger.info("Best command to reproduce:")
    logger.info(
        f"python edge_exp.py "
        f"--dataset {best.get('dataset','cora')} "
        f"--rmax {best.get('rmax')} "
        f"--lam {best.get('lam')} "
        f"--std {best.get('std')} "
        f"--eps {best.get('eps')} "
        f"--prop_step {best.get('prop_step')} "
        f"--weight_mode {best.get('weight_mode')} "
        f"--lr {best.get('lr')} "
        f"--optimizer {best.get('optimizer')} "
        f"--epochs {best.get('epochs')} "
        f"--num_batch_removes 2000 "
        f"--num_removes {best.get('num_removes')} "
        f"--seed {best.get('seed')} "
        f"--dev -1 --disp 100"
    )
    logger.info("=" * 70)

# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ScaleGUN Parameter Tuning")
    parser.add_argument("--mode", default="quick",
                        choices=["paper", "quick", "full", "custom"],
                        help="Which parameter grid to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log all combinations without running")
    args = parser.parse_args()

    grids  = {"paper": PAPER_GRID, "quick": QUICK_GRID,
               "full": FULL_GRID,  "custom": CUSTOM_GRID}
    grid   = grids[args.mode]
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    total  = len(combos)

    logger.info("")
    logger.info("=" * 70)
    logger.info("ScaleGUN Parameter Tuning Started")
    logger.info(f"  Mode      : {args.mode}")
    logger.info(f"  Total runs: {total}")
    logger.info(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Log file  : {LOG_FILE}          ← all output here")
    logger.info(f"  CSV file  : {RESULTS_FILE}   ← results table")
    logger.info(f"  Run logs  : {RUNS_LOG_DIR}/       ← per-run output")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("DRY RUN — All combinations:")
        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            logger.info(f"  {i+1:4d}. " + "  ".join(f"{k}={v}" for k, v in params.items()))
        logger.info(f"\nTotal: {total} combinations")
        return

    est_min = total * 2
    logger.info(f"Estimated time: ~{est_min} min (~{est_min/60:.1f} hrs) at ~2 min/run on CPU")

    if total > 20:
        logger.info(f"Large run ({total} experiments). Starting in 5s... (Ctrl+C to abort)")
        time.sleep(5)

    all_results = []

    for run_id, combo in enumerate(combos, 1):
        params  = dict(zip(keys, combo))
        metrics = run_experiment(params, run_id, total)

        row = {**params, **metrics}
        all_results.append(row)

        # Save after every run — never lose progress
        save_csv(all_results)

        completed_count = sum(
            1 for r in all_results if str(r.get("completed")).lower() == "true"
        )
        logger.info(f"  Progress: {run_id}/{total} done | {completed_count} completed successfully")

    log_summary(all_results)

    logger.info(f"Tuning finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results CSV : {RESULTS_FILE}")
    logger.info(f"Master log  : {LOG_FILE}")
    logger.info("Analyze with: python analyze_results.py")

if __name__ == "__main__":
    main()