#!/usr/bin/env python3
import argparse
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from cartpole import DQL  # Import the DQL function from your cartpole module

# Default constants for the DQL experiments:
N_RUNS = 5
N_TIMESTEPS = 100000
EVAL_INTERVAL = 250
POLICY = "egreedy"
TEMP = 2.0

# Enable the replay buffer and disable the target network.
REPLAY_BUFFER = True
TARGET_NETWORK = False
TARGET_UPDATES = 0  # Not used when target network is off

# Best values for the other hyperparameters:
BEST_CONFIG = {
    "learning_rate": 0.005,
    "gamma": 0.99,
    "hidden_dim": 64,
    "num_hidden_layers": 2,
    "epsilon": 0.2
}

# Candidate values for the replay buffer parameters:
BATCH_SIZE_CANDIDATES = [0.5, 1, 8, 32, 128]
BUFFER_SIZE_CANDIDATES = [32, 128, 1024, 10000]

def run_experiment(config, n_runs=N_RUNS, n_timesteps=N_TIMESTEPS, eval_interval=EVAL_INTERVAL):
    """
    Run the DQL experiment for the given configuration.
    Each run produces an evaluation curve. We compute the normalized AOC (using the trapezoidal rule)
    for each run and return the average normalized AOC along with the individual run scores.
    """
    run_aocs = []
    print("Running experiment with configuration:")
    print(json.dumps(config, indent=4))
    
    for run in range(n_runs):
        start_time = time.time()
        returns, timesteps = DQL(
            n_timesteps,
            config["learning_rate"],
            config["gamma"],
            POLICY,
            config["epsilon"],
            TEMP,
            False,  # Render flag is off
            eval_interval,
            replay_buffer=REPLAY_BUFFER,
            rb_capacity=config["rb_capacity"],
            rb_batch_size=config["rb_batch_size"],
            target_network=TARGET_NETWORK,
            target_updates=TARGET_UPDATES,
            hidden_dim=config["hidden_dim"],
            num_hidden=config["num_hidden_layers"]
        )
        
        elapsed = time.time() - start_time
        print(f"Run {run+1}/{n_runs} completed in {elapsed:.2f} sec")
        
        # Compute the Area Under the Curve (AOC) using the trapezoidal rule.
        aoc = np.trapz(returns, x=timesteps)
        # Assume maximum possible return is 500 per timestep.
        max_possible_area = 500 * timesteps[-1]
        normalized_aoc = aoc / max_possible_area
        run_aocs.append(normalized_aoc)
    
    avg_aoc = np.mean(run_aocs)
    print(f"Average normalized AOC: {avg_aoc:.4f}\n")
    return avg_aoc, run_aocs

def save_results(results, results_path):
    """Save the tuning results dictionary to a JSON file."""
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print("Error saving results:", e)

def load_results(results_path):
    """Load previous tuning results if the file exists, otherwise return an empty dict."""
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print("Error loading results:", e)
            return {}
    else:
        return {}

def plot_marginalized(batch_candidates, buffer_candidates, aoc_matrix, results_dir):
    """
    Create two 2D plots:
      1. Normalized AOC vs. Batch Size (averaged over buffer sizes)
      2. Normalized AOC vs. Buffer Size (averaged over batch sizes)
    """
    # Marginalize over buffer sizes (average across rows)
    marginal_batch = np.nanmean(aoc_matrix, axis=0)
    plt.figure()
    plt.plot(batch_candidates, marginal_batch, marker='o')
    plt.xlabel('Batch Size (rb_batch_size)')
    plt.ylabel('Normalized AOC')
    plt.title('Normalized AOC vs. Batch Size (Marginalized over Buffer Size)')
    plt.grid(True)
    batch_plot_path = os.path.join(results_dir, "marginal_batch_size.png")
    plt.savefig(batch_plot_path)
    plt.close()
    print(f"Saved marginalized batch size plot to {batch_plot_path}")
    
    # Marginalize over batch sizes (average across columns)
    marginal_buffer = np.nanmean(aoc_matrix, axis=1)
    plt.figure()
    plt.plot(buffer_candidates, marginal_buffer, marker='o')
    plt.xlabel('Buffer Size (rb_capacity)')
    plt.ylabel('Normalized AOC')
    plt.title('Normalized AOC vs. Buffer Size (Marginalized over Batch Size)')
    plt.grid(True)
    buffer_plot_path = os.path.join(results_dir, "marginal_buffer_size.png")
    plt.savefig(buffer_plot_path)
    plt.close()
    print(f"Saved marginalized buffer size plot to {buffer_plot_path}")

def plot_3d_surface(batch_candidates, buffer_candidates, aoc_matrix, results_dir):
    """
    Create a 3D surface plot where:
      - x-axis: Batch Size (rb_batch_size)
      - z-axis: Buffer Size (rb_capacity)
      - y-axis: Normalized AOC
    """
    X, Z = np.meshgrid(batch_candidates, buffer_candidates)
    Y = aoc_matrix  # Shape: (len(buffer_candidates), len(batch_candidates))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot a surface. Note: since candidates are discrete, you can also use a scatter plot.
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Batch Size (rb_batch_size)')
    ax.set_ylabel('Normalized AOC')
    ax.set_zlabel('Buffer Size (rb_capacity)')
    ax.set_title('3D Surface Plot for Replay Buffer Tuning')
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Normalized AOC')
    
    plot_path = os.path.join(results_dir, "replay_buffer_tuning_3d.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved 3D surface plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Grid search for replay buffer parameters with soft-starting")
    parser.add_argument("--results_dir", type=str, default="grid_search_replay_buffer_results",
                        help="Directory for storing tuning results and plots")
    parser.add_argument("--n_runs", type=int, default=N_RUNS, help="Number of runs per configuration")
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "grid_search_replay_buffer_results.json")
    
    # Load previous results if available.
    results = load_results(results_path)
    
    # Prepare a matrix to hold normalized AOC values.
    num_buf = len(BUFFER_SIZE_CANDIDATES)
    num_batch = len(BATCH_SIZE_CANDIDATES)
    aoc_matrix = np.empty((num_buf, num_batch))
    aoc_matrix[:] = np.nan  # initialize with NaNs
    
    # Grid search over all combinations.
    for i, buf in enumerate(BUFFER_SIZE_CANDIDATES):
        buf_key = str(buf)
        if buf_key not in results:
            results[buf_key] = {}
        for j, batch in enumerate(BATCH_SIZE_CANDIDATES):
            batch_key = str(batch)
            if batch_key in results[buf_key]:
                score = results[buf_key][batch_key]["normalized_aoc"]
                print(f"Candidate rb_capacity={buf}, rb_batch_size={batch} already evaluated: Score = {score:.4f}")
            else:
                # Build configuration from BEST_CONFIG plus replay buffer settings.
                config = BEST_CONFIG.copy()
                config["rb_capacity"] = buf
                config["rb_batch_size"] = batch
                print("Testing configuration:")
                print(json.dumps(config, indent=4))
                
                score, individual_runs = run_experiment(config, n_runs=args.n_runs)
                results[buf_key][batch_key] = {
                    "normalized_aoc": score,
                    "individual_aocs": individual_runs,
                    "config": config
                }
                save_results(results, results_path)
                print(f"Saved results for rb_capacity={buf}, rb_batch_size={batch}")
            aoc_matrix[i, j] = results[buf_key][batch_key]["normalized_aoc"]
    
    # Generate marginalized plots.
    plot_marginalized(BATCH_SIZE_CANDIDATES, BUFFER_SIZE_CANDIDATES, aoc_matrix, args.results_dir)
    # Generate the 3D surface plot.
    plot_3d_surface(BATCH_SIZE_CANDIDATES, BUFFER_SIZE_CANDIDATES, aoc_matrix, args.results_dir)
    
if __name__ == '__main__':
    main()
