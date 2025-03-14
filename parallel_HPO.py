import itertools
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from cartpole import DQL

# ===============================
# 🚀 Helper Function for Parallel Runs
# ===============================
def run_dql(args):
    """ Wrapper function to run DQL in parallel. """
    (n_timesteps, learning_rate, gamma, policy, epsilon, temp, eval_interval, 
     replay_buffer, rb_capacity, rb_batch_size, target_network, target_updates) = args
    returns, _ = DQL(n_timesteps, learning_rate, gamma, policy, epsilon, temp, False, eval_interval,
                     replay_buffer=replay_buffer, rb_capacity=rb_capacity, rb_batch_size=rb_batch_size,
                     target_network=target_network, target_updates=target_updates)
    return returns

# ===============================
# 🔥 Hyperparameter Optimization (HPO)
# ===============================
def HPO_vanilla():
    n_runs = 10
    n_timesteps = 5000
    eval_interval = 250

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

    total_combinations = len(list(itertools.product(
        gamma_values, learning_rate_values, hidden_dim_values,
        num_hidden_layers_values, policy_values, epsilon_values, temp_values
    )))

    for iteration, (gamma, lr, hidden_dim, num_layers, policy, epsilon, temp) in enumerate(itertools.product(
        gamma_values, learning_rate_values, hidden_dim_values,
        num_hidden_layers_values, policy_values, epsilon_values, temp_values)):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            args_list = [(n_timesteps, lr, gamma, policy, epsilon, temp, eval_interval, 
                          False, None, None, False, None) for _ in range(n_runs)]
            all_returns = list(executor.map(run_dql, args_list))

        mean_returns = np.mean(all_returns, axis=0)
        avg_return = np.mean(mean_returns[-10:])

        params = {'gamma': gamma, 'learning_rate': lr, 'hidden_dim': hidden_dim, 
                  'num_hidden_layers': num_layers, 'policy': policy, 'epsilon': epsilon, 
                  'temp': temp, 'avg_return': avg_return}
        results.append(params)

        if avg_return > best_score:
            best_score = avg_return
            best_params = params

        print(f"{(iteration / total_combinations) * 100:.1f}% done {iteration+1}/{total_combinations}")

    print("\n🔥 Best Hyperparameters 🔥")
    print(best_params)

    pd.DataFrame(results).to_csv("grid_search_results.csv", index=False)
    print("Results saved to grid_search_results.csv")

# ===============================
# 🔥 Tune Replay Buffer
# ===============================
def Tune_Replay_Buffer():
    n_runs = 10
    n_timesteps = 5000
    eval_interval = 250

    capacity_values = [500, 1000, 2000, 5000]
    batch_size_values = [4, 8, 16, 32]

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

    for iteration, (capacity, batch_size) in enumerate(itertools.product(capacity_values, batch_size_values)):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            args_list = [(n_timesteps, learning_rate, gamma, policy, epsilon, temp, eval_interval,
                          True, capacity, batch_size, False, None) for _ in range(n_runs)]
            all_returns = list(executor.map(run_dql, args_list))

        mean_returns = np.mean(all_returns, axis=0)
        avg_return = np.mean(mean_returns[-10:])
        variance = np.var(mean_returns[-10:])

        params = {'capacity': capacity, 'batch_size': batch_size, 'avg_return': avg_return, 'variance': variance}
        results.append(params)

        if avg_return > best_score:
            best_score = avg_return
            best_params = params

        print(f"{(iteration / total_combinations) * 100:.1f}% done {iteration+1}/{total_combinations}")

    print("\n🔥 Best Replay Buffer Configuration 🔥")
    print(best_params)

    pd.DataFrame(results).to_csv("replay_buffer_tuning_results.csv", index=False)
    print("Results saved to replay_buffer_tuning_results.csv")

# ===============================
# 🔥 Tune Target Network
# ===============================
def Tune_Target_Network():
    n_runs = 10
    n_timesteps = 5000
    eval_interval = 250

    target_update_values = [10, 50, 100, 200, 500, 1000]

    gamma = 0.9
    learning_rate = 0.01
    hidden_dim = 8
    num_hidden_layers = 4
    policy = "softmax"
    epsilon = 0.1
    temp = 2.0

    capacity = 5000
    batch_size = 32
    replay_buffer = False

    best_score = float('-inf')
    best_params = None
    results = []

    for iteration, target_updates in enumerate(target_update_values):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            args_list = [(n_timesteps, learning_rate, gamma, policy, epsilon, temp, eval_interval,
                          replay_buffer, capacity, batch_size, True, target_updates) for _ in range(n_runs)]
            all_returns = list(executor.map(run_dql, args_list))

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

    pd.DataFrame(results).to_csv("target_network_tuning_results.csv", index=False)
    print("Results saved to target_network_tuning_results.csv")

# ===============================
# ✅ Run the Selected Function
# ===============================
if __name__ == "__main__":
    # Uncomment one of the following to run the tuning process
    HPO_vanilla()
    # Tune_Replay_Buffer()
    # Tune_Target_Network()
