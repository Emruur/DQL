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

# Parameters for the "Naive" configuration:
REPLAY_BUFFER = False
TARGET_NETWORK = False
RB_CAPACITY = 5000
BATCH_SIZE = 32
TARGET_UPDATES = 100

# Candidate hyperparameter values to be optimized:
CANDIDATE_VALUES = {
    "learning_rate": [0.001, 0.005, 0.01, 0.05],
    "gamma": [0.8, 0.9, 0.95, 0.99],
    "hidden_dim": [16, 32, 64, 128],
    "num_hidden_layers": [1, 2, 3],
    "epsilon": [0.05, 0.1, 0.2]
}

# Default best configuration (will be updated sequentially)
BEST_CONFIG = {
    "learning_rate": 0.01,
    "gamma": 0.9,
    "hidden_dim": 32,
    "num_hidden_layers": 2,
    "epsilon": 0.1
}

def run_experiment(config, n_runs=N_RUNS, n_timesteps=N_TIMESTEPS, eval_interval=EVAL_INTERVAL):
    """
    Run the DQL experiment for the given hyperparameter configuration.
    Each run produces an evaluation curve. We compute the normalized AOC for each run
    and return the average normalized AOC (along with the individual values).
    """
    run_aocs = []
    # Print the full configuration for debugging
    print("Running experiment with configuration:")
    print(json.dumps(config, indent=4))
    
    for run in range(n_runs):
        start_time = time.time()
        # Call the actual DQL function
        returns, timesteps = DQL(n_timesteps,
                                 config["learning_rate"],
                                 config["gamma"],
                                 POLICY,
                                 config["epsilon"],
                                 TEMP,
                                 False,
                                 eval_interval,
                                 replay_buffer=REPLAY_BUFFER,
                                 rb_capacity=RB_CAPACITY,
                                 rb_batch_size=BATCH_SIZE,
                                 target_network=TARGET_NETWORK,
                                 target_updates=TARGET_UPDATES,
                                 hidden_dim=config["hidden_dim"],
                                 num_hidden=config["num_hidden_layers"])
        
        elapsed = time.time() - start_time
        print(f"Run {run+1}/{n_runs} completed in {elapsed:.2f} sec")
        
        # Compute the Area Under the Curve (AOC) using the trapezoidal rule.
        aoc = np.trapz(returns, x=timesteps)
        # Assume maximum possible return is 500 per timestep (adjust if necessary)
        max_possible_area = 500 * timesteps[-1]
        normalized_aoc = aoc / max_possible_area
        run_aocs.append(normalized_aoc)
    
    avg_aoc = np.mean(run_aocs)
    print(f"Average normalized AOC: {avg_aoc:.4f}\n")
    return avg_aoc, run_aocs

def save_results(results, results_path):
    """Robustly save the HPO results dictionary to a JSON file."""
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print("Error saving results:", e)

def load_results(results_path):
    """Load the HPO results if the file exists, otherwise return an empty dict."""
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print("Error loading results:", e)
            return {}
    else:
        return {}

def plot_trend(hp_name, candidate_list, scores, results_dir):
    """Plot and save the performance trend for a given hyperparameter."""
    plt.figure()
    plt.plot(candidate_list, scores, marker='o')
    plt.xlabel(hp_name)
    plt.ylabel('Normalized AOC')
    plt.title(f'Performance trend for {hp_name}')
    plt.grid(True)
    plot_path = os.path.join(results_dir, f"{hp_name}_trend.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved performance trend plot for {hp_name} to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="HPO for Deep Q-Learning on CartPole")
    parser.add_argument("--results_dir", type=str, default="hpo_results",
                        help="Directory for storing HPO results and plots")
    parser.add_argument("--n_runs", type=int, default=N_RUNS, help="Number of runs per configuration")
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "hpo_results.json")
    
    # Load previous HPO results for soft-starting
    hpo_results = load_results(results_path)
    
    # Initialize HPO results for each hyperparameter if not present
    for hp in CANDIDATE_VALUES.keys():
        if hp not in hpo_results:
            hpo_results[hp] = {"tested_values": {}, "best_value": None}
    
    # Sequentially tune hyperparameters in the following order:
    hyperparams_order = ["learning_rate", "gamma", "hidden_dim", "num_hidden_layers", "epsilon"]
    
    # The best configuration so far (initially set to defaults)
    current_best_config = BEST_CONFIG.copy()
    
    for hp in hyperparams_order:
        print(f"Optimizing hyperparameter: {hp}")
        candidate_list = CANDIDATE_VALUES[hp]
        scores = []  # to record average AOC per candidate
        # Evaluate each candidate value if not already done.
        for candidate in candidate_list:
            candidate_str = str(candidate)
            if candidate_str in hpo_results[hp]["tested_values"]:
                # Already evaluated
                score = hpo_results[hp]["tested_values"][candidate_str]["normalized_aoc"]
                print(f"Candidate {hp}={candidate} already evaluated: Score = {score:.4f}")
            else:
                # Update configuration: use the current best for tuned hyperparameters
                # and candidate for the hyperparameter being optimized.
                config = current_best_config.copy()
                config[hp] = candidate
                # Print full configuration for debugging.
                print("Testing configuration:")
                print(json.dumps(config, indent=4))
                # Run the experiment
                score, individual_runs = run_experiment(config, n_runs=args.n_runs)
                # Save the result for this candidate
                hpo_results[hp]["tested_values"][candidate_str] = {
                    "normalized_aoc": score,
                    "individual_aocs": individual_runs,
                    "config": config
                }
                # Save after each candidate to enable soft-starting.
                save_results(hpo_results, results_path)
            scores.append(hpo_results[hp]["tested_values"][candidate_str]["normalized_aoc"])
        
        # Select the candidate with the highest score.
        best_idx = np.argmax(scores)
        best_candidate = candidate_list[best_idx]
        best_score = scores[best_idx]
        hpo_results[hp]["best_value"] = best_candidate
        current_best_config[hp] = best_candidate
        
        # Plot performance trend for the current hyperparameter.
        plot_trend(hp, candidate_list, scores, args.results_dir)
        
        print(f"Best value for {hp}: {best_candidate} (Score: {best_score:.4f})\n")
        # Save updated results.
        save_results(hpo_results, results_path)
    
    print("Final best hyperparameter configuration:")
    print(json.dumps(current_best_config, indent=4))
    
if __name__ == '__main__':
    main()
