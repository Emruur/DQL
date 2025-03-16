import gymnasium as gym
import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import random


def DQL(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot= True,eval_interval=500, replay_buffer= False, rb_capacity= 1000, rb_batch_size= 4, target_network= False, target_updates= 100, hidden_dim= 16, num_hidden= 1, update_data_ratio= 1):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    if plot:
        env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)
    else:
        env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=500)

    eval_env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=500)
    agent= Agent(learning_rate=learning_rate, gamma=gamma, replay_buffer=replay_buffer, rb_capacity= rb_capacity, rb_batch_size= rb_batch_size, target_network=target_network, target_updates=target_updates, num_layers= num_hidden, hidden_dim= hidden_dim)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here!
    s, info = env.reset()
    experiences= []
    for i in range(n_timesteps):
        a= agent.select_action(s, policy, epsilon,temp)
        s_next, r, done, truncated, info = env.step(a)
        finalized= done or truncated
        agent.update(s,a,r,s_next, finalized)
        s= s_next
        if finalized:
            s, info = env.reset()

        if i%eval_interval == 0:
            mean_reward= agent.evaluate(eval_env, n_eval_episodes=30)
            eval_returns.append(mean_reward)
            eval_timesteps.append(i)
    
        if plot:
            env.render()

    env.close()
    eval_env.close()
    return np.array(eval_returns), np.array(eval_timesteps) 

import argparse
import os
import time
import numpy as np
import pandas as pd
# from your_dql_module import DQL  # Ensure that DQL is imported from your module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, choices=["Naive", "Experience Replay", "Target Network", "Both"],
                        help="Select the configuration to run. If not provided, all configurations will be run.")
    args = parser.parse_args()

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    n_runs = 5
    n_timesteps = 1000000
    eval_interval = 250

    # Initial hyperparameters (overwritten below)
    gamma = 0.9
    learning_rate = 0.01
    hidden_dim = 32
    num_hidden_layers = 2
    policy = "egreedy"
    epsilon = 0.1

    temp = 20

    # Updated hyperparameters
    learning_rate = 0.005
    gamma = 0.99
    hidden_dim = 64
    num_hidden_layers = 2
    epsilon = 0.2

    ## Replay Buffer Params
    capacity = 32
    batch_size = 8

    ## Target Q Params
    target_updates = 25

    configs = {
        "Naive": (False, False),
        "Experience Replay": (True, False),
        "Target Network": (False, True),
        "Both": (True, True),
    }

    # If a specific config is provided via command line, only run that one; otherwise, run all.
    selected_configs = {args.config: configs[args.config]} if args.config else configs

    for config_name, (replay_buffer, target_network) in selected_configs.items():
        print(f"\nProcessing configuration: {config_name}")
        run_data_list = []
        all_timesteps = None

        # Loop over each run for the configuration
        for run in range(1, n_runs + 1):
            run_csv_filename = os.path.join(results_dir, f"{config_name.replace(' ', '_').lower()}_run{run}.csv")
            if os.path.exists(run_csv_filename):
                print(f"CSV file for config '{config_name}' run {run} exists, skipping training for this run.")
                df = pd.read_csv(run_csv_filename)
                run_data_list.append(df["return"].to_numpy())
                if all_timesteps is None:
                    all_timesteps = df["timesteps"].to_numpy()
                continue

            start_time = time.time()
            # Run the DQL training for the given configuration and run.
            returns, timesteps = DQL(n_timesteps, learning_rate, gamma, policy, epsilon, temp,
                                       False, eval_interval, replay_buffer=replay_buffer,
                                       rb_capacity=capacity, rb_batch_size=batch_size,
                                       target_network=target_network, target_updates=target_updates,
                                       hidden_dim=hidden_dim, num_hidden=num_hidden_layers)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Run {run} for config '{config_name}' completed in {elapsed_time:.2f} seconds")

            # Save the CSV file for this run.
            df_run = pd.DataFrame({"timesteps": timesteps, "return": returns})
            df_run.to_csv(run_csv_filename, index=False)
            run_data_list.append(np.array(returns))
            if all_timesteps is None:
                all_timesteps = np.array(timesteps)

        # After all runs are completed (or skipped), create a combined CSV.
        combined_csv_filename = os.path.join(results_dir, f"{config_name.replace(' ', '_').lower()}_combined.csv")
        if os.path.exists(combined_csv_filename):
            print(f"Combined CSV for config '{config_name}' exists, skipping combined generation.")
        else:
            run_data_array = np.array(run_data_list)  # Shape: (n_runs, len(timesteps))
            mean_returns = np.mean(run_data_array, axis=0)
            std_returns = np.std(run_data_array, axis=0)
            combined_data = {"timesteps": all_timesteps}
            for i in range(n_runs):
                combined_data[f"run_{i+1}"] = run_data_array[i]
            combined_data["mean_return"] = mean_returns
            combined_data["std_return"] = std_returns
            df_combined = pd.DataFrame(combined_data)
            df_combined.to_csv(combined_csv_filename, index=False)
            print(f"Saved combined results for config '{config_name}' to {combined_csv_filename}")

if __name__ == "__main__":
    main()

