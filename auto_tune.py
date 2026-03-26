import subprocess
import itertools
import re
import time
import csv

# --- GRID SEARCH PARAMETERS ---
rmax_vals = [1e-7, 1e-9, 1e-10]
lam_vals = [1e-4]
std_vals = [0.1, 0.05,0.01]

# Added --disp 1 to ensure the logs are ALWAYS printed
FIXED_ARGS = [
    "python", "edge_exp.py",
    "--dataset", "cora",
    "--num_batch_removes", "10", 
    "--num_removes", "10",       
    "--weight_mode", "test",
    "--layer", "2",
    "--prop_step", "2",
    "--epochs", "250",           
    "--dev", "-1",               
    "--verbose",
    "--disp", "1"  
]

def run_experiment(rmax, lam, std):
    cmd = FIXED_ARGS + [
        "--rmax", str(rmax),
        "--lam", str(lam),
        "--std", str(std)
    ]
    
    print(f"\n🚀 Running: rmax={rmax}, lam={lam}, std={std}")
    start_time = time.time()
    
    # Run the process and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output, _ = process.communicate()
    
    execution_time = time.time() - start_time
    
    # --- BULLETPROOF REGEX PARSING ---
    # 1. Initial Test Accuracy
    init_acc_match = re.search(r"Test accuracy:\s*(\d+\.\d+)", output)
    init_acc = float(init_acc_match.group(1)) if init_acc_match else 0.0
    
    # 2. Final Test Accuracy & Retrains
    # Using DOTALL and handling scientific notation (e.g., 1.5e-04)
    iter_matches = list(re.finditer(r"Test acc\s*=\s*(\d+\.\d+).*?unlearn cost:\s*(\d+\.\d+(?:e[-+]\d+)?).*?num_retrain:\s*(\d+)", output, re.IGNORECASE | re.DOTALL))
    
    if iter_matches:
        last_match = iter_matches[-1]
        final_acc = float(last_match.group(1))
        avg_unlearn_cost = float(last_match.group(2))
        total_retrains = int(last_match.group(3))
    else:
        final_acc, avg_unlearn_cost, total_retrains = 0.0, 999.0, 999
        print("⚠️ DEBUG: Failed to parse logs! Here are the last 500 characters of the raw output:")
        print("-" * 50)
        print(output[-500:])
        print("-" * 50)

    print(f"   ↳ Init Acc: {init_acc:.4f} | Final Acc: {final_acc:.4f} | Retrains: {total_retrains} | Unlearn Cost: {avg_unlearn_cost:.4f}s")
    
    return {
        "rmax": rmax,
        "lam": lam,
        "std": std,
        "init_acc": init_acc,
        "final_acc": final_acc,
        "retrains": total_retrains,
        "unlearn_cost": avg_unlearn_cost,
        "exec_time": execution_time
    }

def main():
    results = []
    param_combinations = list(itertools.product(rmax_vals, lam_vals, std_vals))
    print(f"Starting Hyperparameter Tuning: {len(param_combinations)} combinations to test...")
    
    for rmax, lam, std in param_combinations:
        res = run_experiment(rmax, lam, std)
        results.append(res)
        
    valid_results = [r for r in results if r["retrains"] == 0]
    
    if not valid_results:
        print("\n⚠️ WARNING: No configuration achieved 0 retrains. The budget is still being breached.")
        best_config = sorted(results, key=lambda x: x["final_acc"], reverse=True)[0]
    else:
        valid_results.sort(key=lambda x: (x["final_acc"], -x["unlearn_cost"]), reverse=True)
        best_config = valid_results[0]
        
    print("\n" + "="*50)
    print("🏆 BEST CONFIGURATION IDENTIFIED 🏆")
    print("="*50)
    print(f"  --rmax {best_config['rmax']}")
    print(f"  --lam  {best_config['lam']}")
    print(f"  --std  {best_config['std']}")
    print("-" * 50)
    print(f"Initial Accuracy : {best_config['init_acc']:.4f}")
    print(f"Final Accuracy   : {best_config['final_acc']:.4f}")
    print(f"Total Retrains   : {best_config['retrains']}")
    print(f"Avg Unlearn Cost : {best_config['unlearn_cost']:.4f} seconds")
    print("="*50)
    
    csv_file = "tuning_results_exp3_exp4.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"📊 Full results saved to {csv_file}")

if __name__ == "__main__":
    main()