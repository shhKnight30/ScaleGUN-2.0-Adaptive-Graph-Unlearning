import os
import glob
import re
import csv
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. DEFINE HYPERPARAMETER GRID
# ==========================================
eps_list = [0.4, 0.484, 0.6, 0.8]
std_list = [0.05, 0.07, 0.1]
rmax_list = [1e-7, 1e-8, 1e-9]

csv_filename = "scale_gun_tuning_results.csv"
log_dir = os.path.join("analysis", "cora", "edge")


# ==========================================
# HELPER: GET NEW LOG FILE
# ==========================================
def get_new_log_file(before_logs, after_logs):
    new_logs = list(after_logs - before_logs)
    if not new_logs:
        return None
    # If multiple, pick latest among NEW ones only
    return max(new_logs, key=os.path.getmtime)


# ==========================================
# RUN EXPERIMENT
# ==========================================
def run_experiment(eps, std, rmax):
    cmd = [
        "python", "edge_exp.py",
        "--dataset", "cora",
        "--num_batch_removes", "100",
        "--num_removes", "1",
        "--weight_mode", "test",
        "--layer", "2",
        "--prop_step", "2",
        "--epochs", "250",
        "--dev", "-1",
        "--verbose",
        "--disp", "1",
        "--lam", "0.001",
        "--optimizer", "Adam",
        "--eps", str(eps),
        "--std", str(std),
        "--rmax", str(rmax)
    ]

    print(f"\nRunning: eps={eps}, std={std}, rmax={rmax}...")

    # Snapshot logs BEFORE run
    before_logs = set(glob.glob(os.path.join(log_dir, "*.log")))

    start_time = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    exec_time = time.time() - start_time

    # Wait for file write
    time.sleep(2)

    # Snapshot AFTER run
    after_logs = set(glob.glob(os.path.join(log_dir, "*.log")))

    latest_file = get_new_log_file(before_logs, after_logs)

    if not latest_file:
        print("  [!] ERROR: No new log file detected.")
        return default_result(eps, std, rmax, exec_time)

    print(f"  [+] Found log: {os.path.basename(latest_file)}")

    try:
        with open(latest_file, "r", encoding="utf-8", errors="ignore") as f:
            output = f.read()
    except Exception as e:
        print(f"  [!] File read error: {e}")
        return default_result(eps, std, rmax, exec_time)

    # ==========================================
    # PARSING
    # ==========================================
    init_acc, final_acc = 0.0, 0.0
    retrains = None
    unlearn_time = exec_time

    for line in output.split('\n'):
        line_lower = line.lower()

        # Initial accuracy
        if "test accuracy:" in line_lower:
            match = re.search(r'test accuracy:\s*([0-9.]+)', line_lower)
            if match:
                init_acc = float(match.group(1))

        # Final accuracy
        elif "test acc =" in line_lower:
            match = re.search(r'test acc\s*=\s*([0-9.]+)', line_lower)
            if match:
                final_acc = float(match.group(1))

        # Retrains
        elif "num_retrain:" in line_lower:
            match = re.search(r'num_retrain:\s*(\d+)', line_lower)
            if match:
                retrains = int(match.group(1))
                break  # STOP after first valid match

        # Time
        elif "tot cost:" in line_lower and "s" in line_lower:
            match = re.search(r'tot cost:\s*([0-9.]+)', line_lower)
            if match:
                unlearn_time = float(match.group(1))

    if retrains is None:
        print("  [!] Parsing failed: 'num_retrain' not found")
        retrains = 999

    acc_drop = init_acc - final_acc

    return {
        "eps": eps,
        "std": std,
        "rmax": rmax,
        "init_acc": init_acc,
        "final_acc": final_acc,
        "acc_drop": round(acc_drop, 4),
        "retrains": retrains,
        "unlearn_time": round(unlearn_time, 2)
    }


def default_result(eps, std, rmax, exec_time):
    return {
        "eps": eps,
        "std": std,
        "rmax": rmax,
        "init_acc": 0.0,
        "final_acc": 0.0,
        "acc_drop": 0.0,
        "retrains": 999,
        "unlearn_time": round(exec_time, 2)
    }


# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    results = []

    with open(csv_filename, mode='w', newline='') as f:
        fieldnames = ["eps", "std", "rmax", "init_acc", "final_acc", "acc_drop", "retrains", "unlearn_time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for eps in eps_list:
            for std in std_list:
                for rmax in rmax_list:
                    res = run_experiment(eps, std, rmax)
                    results.append(res)
                    writer.writerow(res)
                    f.flush()

                    print(f"  --> Result | Acc Drop: {res['acc_drop']*100:.2f}% | Retrains: {res['retrains']} | Time: {res['unlearn_time']}s")

    print(f"\nData saved to {csv_filename}. Generating plots...")

    # ==========================================
    # PLOTTING
    # ==========================================
    df = pd.DataFrame(results)
    df['rmax_label'] = df['rmax'].apply(lambda x: f"{x:.0e}")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.barplot(data=df, x='eps', y='unlearn_time', hue='rmax_label', ax=axes[0])
    axes[0].set_title('Unlearning Time vs Epsilon')

    sns.lineplot(data=df, x='std', y='acc_drop', hue='eps', marker='o', ax=axes[1])
    axes[1].set_title('Accuracy Drop vs Noise')

    sns.barplot(data=df, x='eps', y='retrains', hue='std', ax=axes[2])
    axes[2].set_title('Retrains vs Epsilon')

    plt.tight_layout()
    plt.savefig("scalegun_tuning_analysis.png", dpi=300)
    print("Plots saved.")
    plt.show()