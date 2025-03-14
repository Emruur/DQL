import gymnasium as gym
import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt
import itertools
from cartpole import DQL
import pandas as pd
import csv




def HPO_vanilla():
    n_runs = 10  # Run each setting 10 times for averaging
    n_timesteps = 5000
    eval_interval = 250

    # Define hyperparameter search space
    gamma_values = [0.90, 0.95, 0.99, 1.0]
    learning_rate_values = [0.01, 0.05, 0.1, 0.2]
    hidden_dim_values = [8, 16, 32, 64]
    num_hidden_layers_values = [1, 2, 3, 4]
    policy_values = ['egreedy', 'softmax']
    epsilon_values = [0.05, 0.1, 0.2, 0.3]
    temp_values = [0.5, 1.0, 1.5, 2.0]

    best_score = float('-inf')
    best_params = None
    results = []

    total_combinations = (len(gamma_values) * len(learning_rate_values) *
                          len(hidden_dim_values) * len(num_hidden_layers_values) *
                          len(policy_values) * len(epsilon_values) * len(temp_values))
    iteration = 0

    for gamma, lr, hidden_dim, num_layers, policy, epsilon, temp in itertools.product(
            gamma_values, learning_rate_values, hidden_dim_values,
            num_hidden_layers_values, policy_values, epsilon_values, temp_values):

        all_returns = []

        # Run multiple times for averaging
        for _ in range(n_runs):
            returns, _ = DQL(n_timesteps, lr, gamma, policy, epsilon, temp, False, eval_interval)
            all_returns.append(returns)

        # Compute mean performance
        mean_returns = np.mean(all_returns, axis=0)
        avg_return = np.mean(mean_returns[-10:])  # Average of last 10 evaluation points

        params = {
            'gamma': gamma,
            'learning_rate': lr,
            'hidden_dim': hidden_dim,
            'num_hidden_layers': num_layers,
            'policy': policy,
            'epsilon': epsilon,
            'temp': temp,
            'avg_return': avg_return
        }
        results.append(params)

        if avg_return > best_score:
            best_score = avg_return
            best_params = params

        iteration += 1
        print(f"{(iteration / total_combinations) * 100:.1f}% done {iteration}/{total_combinations}")

    print("\n🔥 Best Hyperparameters 🔥")
    print(best_params)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("grid_search_results.csv", index=False)
    print("Results saved to grid_search_results.csv")

def Tune_Replay_Buffer():
    n_runs = 10
    n_timesteps = 5000
    eval_interval = 250

    # Replay Buffer Params to Tune
    capacity_values = [500, 1000, 2000, 5000]
    batch_size_values = [4, 8, 16, 32]

    # Fixed Hyperparameters
    gamma = 0.9
    learning_rate = 0.01
    hidden_dim = 8
    num_hidden_layers = 4
    policy = "softmax"
    epsilon = 0.1
    temp = 2.0

    best_score = float('-inf')
    best_params = None
    results = []

    total_combinations = len(capacity_values) * len(batch_size_values)
    iteration = 0

    for capacity, batch_size in itertools.product(capacity_values, batch_size_values):
        all_returns = []

        for _ in range(n_runs):
            returns, _ = DQL(n_timesteps, learning_rate, gamma, policy, epsilon, temp, False, eval_interval,
                             replay_buffer=True, rb_capacity=capacity, rb_batch_size=batch_size)
            all_returns.append(returns)

        mean_returns = np.mean(all_returns, axis=0)
        avg_return = np.mean(mean_returns[-10:])
        variance = np.var(mean_returns[-10:])

        params = {'capacity': capacity, 'batch_size': batch_size, 'avg_return': avg_return, 'variance': variance}
        results.append(params)

        if avg_return > best_score:
            best_score = avg_return
            best_params = params

        iteration += 1
        print(f"{(iteration / total_combinations) * 100:.1f}% done {iteration}/{total_combinations}")

    print("\n🔥 Best Replay Buffer Configuration 🔥")
    print(best_params)

    # Save results
    pd.DataFrame(results).to_csv("replay_buffer_tuning_results.csv", index=False)
    print("Results saved to replay_buffer_tuning_results.csv")


def Tune_Target_Network():
    n_runs = 10
    n_timesteps = 5000
    eval_interval = 250

    # Target Network Params to Tune
    target_update_values = [10, 50, 100, 200, 500, 1000]

    # Fixed Hyperparameters
    gamma = 0.9
    learning_rate = 0.01
    hidden_dim = 8
    num_hidden_layers = 4
    policy = "softmax"
    epsilon = 0.1
    temp = 2.0

    # Replay Buffer Params
    capacity = 5000
    batch_size = 32
    replay_buffer = False

    best_score = float('-inf')
    best_params = None
    results = []

    for target_updates in target_update_values:
        all_returns = []

        for _ in range(n_runs):
            returns, _ = DQL(n_timesteps, learning_rate, gamma, policy, epsilon, temp, False, eval_interval,
                             replay_buffer=replay_buffer, rb_capacity=capacity, rb_batch_size=batch_size,
                             target_network=True, target_updates=target_updates)
            all_returns.append(returns)

        mean_returns = np.mean(all_returns, axis=0)
        avg_return = np.mean(mean_returns[-10:])
        variance = np.var(mean_returns[-10:])

        params = {'target_updates': target_updates, 'avg_return': avg_return, 'variance': variance}
        results.append(params)

        if avg_return > best_score:
            best_score = avg_return
            best_params = params

        print(f"Processed Target Update {target_updates}: Avg Return = {avg_return:.4f}")

    print("\n🔥 Best Target Network Configuration 🔥")
    print(best_params)

    # Save results
    pd.DataFrame(results).to_csv("target_network_tuning_results.csv", index=False)
    print("Results saved to target_network_tuning_results.csv")

# ✅ Run tuning only if executed directly
if __name__ == "__main__":
    Tune_Target_Network()

