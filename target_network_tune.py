#!/usr/bin/env python3
import argparse
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from cartpole import DQL  # Import the DQL function from your cartpole module

# Default constants for the DQL experiments:
N_RUNS = 5
N_TIMESTEPS = 100000
EVAL_INTERVAL = 250
POLICY = "egreedy"
TEMP = 2.0

# Experience replay buffer is deactivated
REPLAY_BUFFER = False
RB_CAPACITY = 5000
BATCH_SIZE = 32

# We enable the target network and will tune its update frequency.
TARGET_NETWORK = True
TARGET_UPDATES_CANDIDATES = [10,25,50, 100, 200, 500]

# Best values for the rest of the parameters:
BEST_CONFIG = {
    "learning_rate": 0.005,
    "gamma": 0.99,
    "hidden_dim": 64,
    "num_hidden_layers": 2,
    "epsilon": 0.2
}

def run_experiment(config, n_runs=N_RUNS, n_timesteps=N_TIMESTEPS, eval_interval=EVAL_INTERVAL):
    """
    Run the DQL experiment for the given configuration.
    Each run produces an evaluation curve. We compute the normalized AOC for each run
    and return the average normalized AOC along with the individual run scores.
    """
    run_aocs = []
    print("Running experiment with configuration:")
    print(json.dumps(config, indent=4))
    
    for run in range(n_runs):
        start_time = time.time()
        returns, timesteps = DQL(n_timesteps,
                                 config["learning_rate"],
                                 config["gamma"],
                                 POLICY,
                                 config["epsilon"],
                                 TEMP,
                                 False,  # Render flag is False
                                 eval_interval,
                                 replay_buffer=REPLAY_BUFFER,
                                 rb_capacity=RB_CAPACITY,
                                 rb_batch_size=BATCH_SIZE,
                                 target_network=TARGET_NETWORK,
                                 target_updates=config["target_updates"],
                                 hidden_dim=config["hidden_dim"],
                                 num_hidden=config["num_hidden_layers"])
        
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

def plot_trend(candidate_list, scores, results_dir):
    """Plot and save the performance trend for target_updates."""
    plt.figure()
    plt.plot(candidate_list, scores, marker='o')
    plt.xlabel('target_updates')
    plt.ylabel('Normalized AOC')
    plt.title('Performance trend for target_updates')
    plt.grid(True)
    plot_path = os.path.join(results_dir, "target_updates_trend.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved performance trend plot for target_updates to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Tune target_updates for DQL on CartPole with soft-starting")
    parser.add_argument("--results_dir", type=str, default="tune_target_network_results",
                        help="Directory for storing tuning results and plots")
    parser.add_argument("--n_runs", type=int, default=N_RUNS, help="Number of runs per configuration")
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "tune_target_network_results.json")
    
    # Load previous results if available
    results = load_results(results_path)
    scores = []
    
    # Evaluate candidate values for target_updates with soft-starting.
    for candidate in TARGET_UPDATES_CANDIDATES:
        candidate_str = str(candidate)
        if candidate_str in results:
            # Already evaluated: use saved score.
            score = results[candidate_str]["normalized_aoc"]
            print(f"Candidate target_updates={candidate} already evaluated: Score = {score:.4f}")
        else:
            # Prepare configuration for new candidate.
            config = BEST_CONFIG.copy()
            config["target_updates"] = candidate
            print("Testing configuration:")
            print(json.dumps(config, indent=4))
            
            score, individual_runs = run_experiment(config, n_runs=args.n_runs)
            results[candidate_str] = {
                "normalized_aoc": score,
                "individual_aocs": individual_runs,
                "config": config
            }
            save_results(results, results_path)
        scores.append(results[candidate_str]["normalized_aoc"])
    
    # Determine the best candidate.
    best_idx = np.argmax(scores)
    best_candidate = TARGET_UPDATES_CANDIDATES[best_idx]
    best_score = scores[best_idx]
    
    print(f"Best target_updates value: {best_candidate} (Score: {best_score:.4f})\n")
    # Update the performance trend plot after all configurations have been evaluated.
    plot_trend(TARGET_UPDATES_CANDIDATES, scores, args.results_dir)
    
if __name__ == '__main__':
    main()
