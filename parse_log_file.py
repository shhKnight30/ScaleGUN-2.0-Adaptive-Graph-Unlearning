import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIG
# ==========================================
log_dir = "analysis/cora/edge"


# ==========================================
# PARSE SINGLE LOG FILE
# ==========================================
def parse_log(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    filename = os.path.basename(filepath)

    # -------- Extract from filename --------
    std_match = re.search(r'std_([0-9.]+)', filename)
    rmax_match = re.search(r'rmax_([0-9e\-]+)', filename)

    std = float(std_match.group(1)) if std_match else None
    rmax = float(rmax_match.group(1)) if rmax_match else None

    # -------- Extract metrics --------
    init_match = re.search(r'Test accuracy:\s*([0-9.]+)', text, re.IGNORECASE)
    init_acc = float(init_match.group(1)) if init_match else None

    final_matches = re.findall(r'Test acc\s*=\s*([0-9.]+)', text, re.IGNORECASE)
    final_acc = float(final_matches[-1]) if final_matches else None

    retrain_matches = re.findall(r'num_retrain[^0-9]*(\d+)', text, re.IGNORECASE)
    retrains = int(retrain_matches[-1]) if retrain_matches else None

    cost_matches = re.findall(r'tot cost[^0-9]*([0-9.]+)', text, re.IGNORECASE)
    unlearn_time = float(cost_matches[-1]) if cost_matches else None

    acc_drop = (init_acc - final_acc) if init_acc and final_acc else None

    return {
        "file": filename,
        "std": std,
        "rmax": rmax,
        "init_acc": init_acc,
        "final_acc": final_acc,
        "acc_drop": acc_drop,
        "retrains": retrains,
        "unlearn_time": unlearn_time
    }


# ==========================================
# PARSE ALL LOG FILES
# ==========================================
def parse_all_logs(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    print(f"[INFO] Found {len(log_files)} log files")

    data = []
    for file in log_files:
        try:
            res = parse_log(file)
            data.append(res)
        except Exception as e:
            print(f"[ERROR] Failed parsing {file}: {e}")

    df = pd.DataFrame(data)

    # -------- DEBUG --------
    print("\n[DEBUG] Raw DataFrame:")
    print(df.head())
    print("Total rows:", len(df))

    # Only drop rows that break plotting
    df = df.dropna(subset=["std", "rmax", "retrains"])

    print("\n[DEBUG] Cleaned DataFrame:")
    print(df.head())
    print("Remaining rows:", len(df))

    return df


# ==========================================
# PLOTTING (DARK MODE)
# ==========================================
def plot_results(df):
    if df.empty:
        print("[ERROR] DataFrame is empty. Nothing to plot.")
        return

    df['rmax_label'] = df['rmax'].apply(lambda x: f"{x:.0e}")

    # -------- FORCE DARK MODE --------
    plt.style.use("dark_background")
    sns.set_theme(
        style="dark",
        rc={
            "axes.facecolor": "#121212",
            "figure.facecolor": "#121212",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "#444444"
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ensure background applied AFTER subplot creation
    fig.patch.set_facecolor('#121212')
    for ax in axes:
        ax.set_facecolor('#121212')

    # -------- Plot 1 --------
    sns.barplot(data=df, x='std', y='unlearn_time', hue='rmax_label', ax=axes[0])
    axes[0].set_title('Unlearning Time vs Noise (std)')

    # -------- Plot 2 --------
    sns.lineplot(data=df, x='std', y='acc_drop', hue='rmax', marker='o', ax=axes[1])
    axes[1].set_title('Accuracy Drop vs Noise')

    # -------- Plot 3 --------
    sns.barplot(data=df, x='std', y='retrains', hue='rmax_label', ax=axes[2])
    axes[2].set_title('Retrains vs Noise')

    # Grid visibility
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("parsed_log_analysis_dark.png", dpi=300, facecolor='#121212')

    print("[INFO] Plot saved as parsed_log_analysis_dark.png")

    plt.show()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    df = parse_all_logs(log_dir)
    plot_results(df)