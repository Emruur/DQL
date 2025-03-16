import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from cartpole import DQL

def hyperparameter_optimization(results_dir="naive_hpo"):
    # Define search space
    learning_rate_values = [0.001, 0.005, 0.01]
    gamma_values = [0.80, 0.9, 0.99]
    hidden_dim_values = [8, 16, 32]
    num_hidden_layers_values = [1, 2, 4]
    epsilon_values = [0.05, 0.1, 0.2]

    n_runs = 5
    n_timesteps = 100000
    eval_interval = 250
    
    os.makedirs(results_dir, exist_ok=True)
    
    best_params = {
        "learning_rate": 0.01,
        "gamma": 0.9,
        "hidden_dim": 32,
        "num_hidden_layers": 2,
        "epsilon": 0.1
    }
    
    results_file = os.path.join(results_dir, "hpo_results.json")
    
    # Load previous results if available
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            prev_results = json.load(f)
        print("Loaded previous results. Using best known values where available.")
        for key in best_params:
            if key in prev_results:
                best_params[key] = prev_results[key]
    else:
        prev_results = {}

    def evaluate_hyperparam(hyperparam_name, values):
        aoc_results = []
        
        for value in values:
            config = {
                "learning_rate": best_params["learning_rate"],
                "gamma": best_params["gamma"],
                "hidden_dim": best_params["hidden_dim"],
                "num_hidden_layers": best_params["num_hidden_layers"],
                "epsilon": best_params["epsilon"]
            }
            config[hyperparam_name] = value
            print(f"Evaluating {hyperparam_name} with config: {config}")
            
            all_aoc = []
            for run in range(n_runs):
                print(f"  Run {run+1}/{n_runs}")
                returns, timesteps = DQL(n_timesteps, 
                                         config["gamma"],
                                         config["learning_rate"],
                                         "egreedy", config["epsilon"], 2.0,
                                         False, eval_interval,
                                         replay_buffer=False, rb_capacity=5000, rb_batch_size=32,
                                         target_network=False, target_updates=100,
                                         hidden_dim=config["hidden_dim"],
                                         num_hidden=config["num_hidden_layers"])
                
                if timesteps is None or len(timesteps) == 0:
                    print(f"Warning: No timesteps returned for {hyperparam_name}={value}. Skipping AOC computation.")
                    continue
                
                mean_returns = np.mean(returns, axis=0)
                aoc = np.trapezoid(mean_returns, x=timesteps)
                max_possible_area = 500 * timesteps[-1]
                normalized_aoc = aoc / max_possible_area
                all_aoc.append(normalized_aoc)
                
            if all_aoc:
                mean_aoc = np.mean(all_aoc)
                aoc_results.append(mean_aoc)
            else:
                aoc_results.append(0)  # Default to zero if no valid runs
                
        best_value = values[np.argmax(aoc_results)]
        best_params[hyperparam_name] = best_value
        prev_results[hyperparam_name] = best_value
        
        # Save updated best parameters
        with open(results_file, "w") as f:
            json.dump(prev_results, f, indent=4)
        
        # Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(values, aoc_results, marker='o', linestyle='-')
        plt.xlabel(hyperparam_name)
        plt.ylabel("Normalized AOC")
        plt.title(f"{hyperparam_name} Optimization")
        plt.savefig(os.path.join(results_dir, f"hpo_{hyperparam_name}.png"))
        plt.close()

    # Run HPO in a sequential manner
    for param, values in zip(["learning_rate", "gamma", "hidden_dim", "num_hidden_layers", "epsilon"],
                              [learning_rate_values, gamma_values, hidden_dim_values, num_hidden_layers_values, epsilon_values]):
        if param not in prev_results:
            evaluate_hyperparam(param, values)
        else:
            print(f"Skipping {param}, already optimized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="naive_hpo", help="Folder to store/load HPO results")
    args = parser.parse_args()
    hyperparameter_optimization(args.results_dir)
