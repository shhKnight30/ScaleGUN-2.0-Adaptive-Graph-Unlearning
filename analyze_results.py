"""
ScaleGUN Tuning Results Analyzer
==================================
Reads tuning_results.csv and logs analysis to analyze_results.log
Usage: python analyze_results.py
"""

import csv
import os
import sys
import logging
from collections import defaultdict
from datetime import datetime

RESULTS_FILE = "./tuning_results.csv"
ANALYZE_LOG  = "./analyze_results.log"

# ─── LOGGING SETUP ────────────────────────────────────────────────
logger = logging.getLogger("analyzer")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(ANALYZE_LOG, mode="w")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

# ─── LOAD CSV ─────────────────────────────────────────────────────
def load_results():
    if not os.path.exists(RESULTS_FILE):
        logger.error(f"{RESULTS_FILE} not found. Run tune_scalegun.py first.")
        sys.exit(1)

    rows = []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["initial_test_acc", "final_test_acc", "initial_val_acc",
                        "final_val_acc", "avg_tot_cost", "avg_unlearn_cost",
                        "num_retrain", "wall_time", "train_cost"]:
                try:
                    row[key] = float(row[key]) if row[key] not in ("None", "", "null") else None
                except:
                    row[key] = None
            rows.append(row)
    return rows

# ─── PARAMETER IMPACT ─────────────────────────────────────────────
def analyze_parameter_impact(results, param, metric="final_test_acc"):
    groups = defaultdict(list)
    for r in results:
        val = r.get(param)
        m   = r.get(metric)
        if val and m is not None:
            groups[val].append(m)

    summary = []
    for val, vals in sorted(groups.items()):
        summary.append({
            "value" : val,
            "avg"   : round(sum(vals) / len(vals), 4),
            "best"  : round(max(vals), 4),
            "worst" : round(min(vals), 4),
            "count" : len(vals),
        })
    return sorted(summary, key=lambda x: x["avg"], reverse=True)

# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    results   = load_results()
    completed = [r for r in results
                 if str(r.get("completed")).lower() == "true"
                 and r.get("final_test_acc") is not None]

    logger.info("")
    logger.info("=" * 70)
    logger.info("ScaleGUN Tuning Results Analysis")
    logger.info(f"  Analyzed  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Total runs: {len(results)}")
    logger.info(f"  Completed : {len(completed)}")
    logger.info(f"  Failed    : {len(results) - len(completed)}")
    logger.info("=" * 70)

    if not completed:
        logger.error("No completed runs found.")
        return

    # ── TOP 10 BY FINAL TEST ACCURACY ──────────────────────────────
    top10 = sorted(completed, key=lambda x: x.get("final_test_acc", 0), reverse=True)[:10]

    logger.info("")
    logger.info("TOP 10 BY FINAL TEST ACCURACY")
    logger.info(f"  {'#':<4} {'FinalTest':>10} {'InitTest':>10} {'Retrains':>9} "
                f"{'TotCost':>9} {'rmax':<9} {'lam':<7} {'std':<6} {'lr':<7} {'opt':<7} {'seed'}")
    logger.info(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*9} {'─'*9} "
                f"{'─'*9} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*4}")
    for i, r in enumerate(top10):
        logger.info(
            f"  {i+1:<4} "
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

    # ── TOP 10 BY SPEED ────────────────────────────────────────────
    by_speed = sorted(
        [r for r in completed if r.get("avg_tot_cost")],
        key=lambda x: x["avg_tot_cost"]
    )[:10]

    logger.info("")
    logger.info("TOP 10 BY SPEED (lowest avg total cost per edge)")
    logger.info(f"  {'#':<4} {'TotCost':>10} {'FinalTest':>10} {'Retrains':>9} "
                f"{'rmax':<9} {'lam':<7} {'std':<6} {'lr':<7} {'opt'}")
    logger.info(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*9} "
                f"{'─'*9} {'─'*7} {'─'*6} {'─'*7} {'─'*7}")
    for i, r in enumerate(by_speed):
        logger.info(
            f"  {i+1:<4} "
            f"{r.get('avg_tot_cost') or 0:>10.4f} "
            f"{r.get('final_test_acc') or 0:>10.4f} "
            f"{str(r.get('num_retrain', '?')):>9} "
            f"{str(r.get('rmax', '?')):<9} "
            f"{str(r.get('lam', '?')):<7} "
            f"{str(r.get('std', '?')):<6} "
            f"{str(r.get('lr', '?')):<7} "
            f"{str(r.get('optimizer', '?')):<7}"
        )

    # ── ZERO RETRAIN CONFIGS ───────────────────────────────────────
    zero_retrain = [r for r in completed if r.get("num_retrain") == 0.0]
    logger.info("")
    logger.info(f"ZERO-RETRAIN CONFIGS: {len(zero_retrain)} found")
    if zero_retrain:
        best_zero = sorted(zero_retrain, key=lambda x: x.get("final_test_acc", 0), reverse=True)[:5]
        logger.info("  Top 5 by accuracy:")
        for r in best_zero:
            logger.info(
                f"    FinalTest={r.get('final_test_acc'):.4f}  "
                f"rmax={r.get('rmax')}  lam={r.get('lam')}  "
                f"std={r.get('std')}  lr={r.get('lr')}  "
                f"opt={r.get('optimizer')}  seed={r.get('seed')}"
            )

    # ── PARAMETER IMPACT ───────────────────────────────────────────
    logger.info("")
    logger.info("PARAMETER IMPACT ON FINAL TEST ACCURACY")
    logger.info("(average accuracy per parameter value, sorted best → worst)")
    logger.info("")

    for param in ["rmax", "lam", "std", "eps", "prop_step",
                  "weight_mode", "lr", "optimizer", "epochs", "num_removes"]:
        impact = analyze_parameter_impact(completed, param)
        if len(impact) <= 1:
            continue
        logger.info(f"  {param}:")
        for row in impact:
            bar = "█" * int(row["avg"] * 40)
            logger.info(f"    {str(row['value']):>8}  avg={row['avg']:.4f}  "
                        f"best={row['best']:.4f}  worst={row['worst']:.4f}  "
                        f"n={row['count']:>3}  {bar}")
        logger.info("")

    # ── BEST CONFIG ────────────────────────────────────────────────
    best = top10[0]
    logger.info("=" * 70)
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
    logger.info(f"Full analysis saved to: {ANALYZE_LOG}")

if __name__ == "__main__":
    main()