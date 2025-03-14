import gymnasium as gym
import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt
import itertools
import matplotlib.pyplot as plt
import csv

def DQL(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot= True,eval_interval=500, replay_buffer= False, rb_capacity= 1000, rb_batch_size= 4, target_network= False, target_updates= 100):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    if plot:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1", render_mode="rgb_array")

    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent= Agent(learning_rate=learning_rate, gamma=gamma, replay_buffer=replay_buffer, rb_capacity= rb_capacity, rb_batch_size= rb_batch_size, target_network=target_network, target_updates=target_updates)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here!
    s, info = env.reset()
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

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # Create results directory if not exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    n_runs = 10
    n_timesteps = 100000
    eval_interval = 500

    gamma = 0.9
    learning_rate = 0.01
    hidden_dim = 8
    num_hidden_layers = 4
    policy = "softmax"
    epsilon = 0.1
    temp = 2.0

    ## Replay Buffer Params
    capacity = 5000
    batch_size = 32

    ## Target Q Params
    target_updates = 50

    configs = {
        "Naive": (False, False),
        "Experience Replay": (True, False),
        "Target Network": (False, True),
        "Both": (True, True),
    }

    results = {}
    
    for config_name, (replay_buffer, target_network) in configs.items():
        all_returns = []
        all_timesteps = None  # assume timesteps are consistent across runs

        for i in range(n_runs):
            returns, timesteps = DQL(n_timesteps, learning_rate, gamma, policy, epsilon, temp, 
                                     False, eval_interval, replay_buffer=replay_buffer, 
                                     rb_capacity=capacity, rb_batch_size=batch_size, 
                                     target_network=target_network, target_updates=target_updates)
            all_returns.append(returns)
            print(f"run {i} config {config_name}")
            if all_timesteps is None:
                all_timesteps = timesteps

        all_returns = np.array(all_returns)  # shape: (n_runs, len(timesteps))

        # Compute the mean and standard deviation (for each timestep) across runs
        mean_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)
        results[config_name] = (all_timesteps, mean_returns, std_returns)
        
        # Compute Area Under the Curve (AOC) using the trapezoidal rule
        aoc = np.trapz(mean_returns, x=all_timesteps)
        # Normalize the AOC given y ranges from 0 to 150 and x ranges from 0 to last timestep
        max_possible_area = 150 * all_timesteps[-1]
        normalized_aoc = aoc / max_possible_area

        # Plot individual graphs
        plt.figure(figsize=(10, 6))
        plt.plot(all_timesteps, mean_returns, label="Mean Return")
        plt.fill_between(all_timesteps, mean_returns - std_returns, mean_returns + std_returns,
                         color='gray', alpha=0.3, label="± 1 STD")
        plt.xlabel("Timesteps")
        plt.ylabel("Evaluation Return")
        plt.title(f"{config_name} - Mean Evaluation Return with Std Bound (Max: {max(mean_returns):.2f}, Norm AOC: {normalized_aoc:.2f})")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{config_name.replace(' ', '_').lower()}.png"))
        plt.close()

    # Plot all means in a single graph (Graph 5)
    plt.figure(figsize=(10, 6))
    for config_name, (timesteps, mean_returns, _) in results.items():
        plt.plot(timesteps, mean_returns, label=config_name)

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Return")
    plt.title("Comparison of Mean Evaluation Returns Across Configurations")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "comparison.png"))
    plt.close()



# n_timesteps = 50000
# eval_interval = 250

# gamma = 0.9
# learning_rate = 0.01
# hidden_dim = 8
# num_hidden_layers = 4
# policy = "softmax"
# epsilon = 0.1
# temp = 2.0

# ## Replay Buffer Params
# capacity = 5000
# batch_size = 32

# ## Target Q Params
# target_updates = 50


# returns, timesteps= DQL(n_timesteps, learning_rate, gamma, policy, epsilon, temp, 
#                                      False, eval_interval, replay_buffer=True, 
#                                      rb_capacity=capacity, rb_batch_size=batch_size, 
#                                      target_network=True, target_updates=target_updates)

# plt.figure(figsize=(10, 5))
# plt.plot(timesteps, returns, marker='o', linestyle='-', label='DQL Returns')
# plt.xlabel('Timesteps')
# plt.ylabel('Returns')
# plt.title('DQL Returns Over Timesteps')
# plt.legend()
# plt.grid()
# plt.show()