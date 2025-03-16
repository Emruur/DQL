import numpy as np
from Policy import CartPoleQ

import numpy as np
import torch

import numpy as np
import random
from collections import deque

import copy

class ReplayBuffer:
    def __init__(self, capacity=1000, batch_size=4):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        # If batch_size is a fractional value, treat it as a probability to return one sample.
        if self.batch_size < 1:
            if random.random() < self.batch_size:
                batch = random.sample(self.buffer, 1)
            else:
                return None  # or return (np.array([]), np.array([]), np.array([], dtype=np.float32), np.array([]), np.array([], dtype=np.uint8))
        else:
            batch = random.sample(self.buffer, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )


    def ready(self):

        return len(self.buffer) >= self.batch_size

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, learning_rate, gamma, n_actions=2, num_layers= 2, hidden_dim= 16, replay_buffer= False, rb_capacity= 1000, rb_batch_size= 4, target_network= False, target_updates= 100):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_actions = n_actions
        self.use_replay_buffer= replay_buffer
        self.use_target_network= target_network
        # Initialize the Q-network that takes a 5-dim input and outputs a scalar Q-value.
        self.Q_sa = CartPoleQ(lr=learning_rate, hidden_dim= hidden_dim, num_hidden_layers= num_layers)
        if target_network:
            self.target_Q_sa = CartPoleQ(lr=learning_rate, hidden_dim= hidden_dim, num_hidden_layers= num_layers)
            self.target_Q_sa.load_state_dict(copy.deepcopy(self.Q_sa.state_dict()))
        if replay_buffer:
            self.buffer= ReplayBuffer(rb_capacity, rb_batch_size)

        self.update_count= 0
        self.target_updates= target_updates
    
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        # For each action in {0, 1} create an input vector of shape (5,)
        input_batch = np.stack([np.concatenate((s, [a])) for a in range(self.n_actions)], axis=0)
        input_tensor = torch.tensor(input_batch, dtype=torch.float32)
        
        # Forward pass in batch: results will be of shape [n_actions, 1]
        q_values_tensor = self.Q_sa.forward(input_tensor)
        # Squeeze to get shape (n_actions,) and convert to numpy array
        q_values = q_values_tensor.squeeze().detach().numpy()
        
        if policy == 'greedy':
            a = int(np.argmax(q_values))
        
        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon for e-greedy policy")
            if np.random.rand() >= epsilon:
                a = int(np.argmax(q_values))
            else:
                a = np.random.randint(0, self.n_actions)
        
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature for softmax policy")
            # For numerical stability subtract max(q_values)
            stable_q = q_values - np.max(q_values)
            exp_q = np.exp(stable_q / temp)
            sum_exp_q = np.sum(exp_q)
            if sum_exp_q == 0 or np.isnan(sum_exp_q):
                a_p = np.ones(self.n_actions) / self.n_actions
            else:
                a_p = exp_q / sum_exp_q
            a = int(np.random.choice(self.n_actions, p=a_p))
        
        else:
            raise ValueError("Unsupported policy type")
        
        return a
    
    def query_Q(self,input_tensor):
        if self.use_target_network:
            return self.target_Q_sa.forward(input_tensor)
        return self.Q_sa.forward(input_tensor)
    
    def update_target_Q(self):
        self.target_Q_sa.load_state_dict(copy.deepcopy(self.Q_sa.state_dict()))

    def update(self, s, a, r, s_next, done):
        
        self.update_count += 1
        if self.use_target_network:
            if self.update_count % self.target_updates== 0:
                self.update_target_Q()
        
        if not self.use_replay_buffer:
            # Compute target for single experience
            target = r
            if not done:
                input_batch = np.stack([np.concatenate((s_next, [action])) for action in range(self.n_actions)], axis=0)
                input_tensor = torch.tensor(input_batch, dtype=torch.float32)
                q_values_tensor = self.query_Q(input_tensor)
                q_values = q_values_tensor.squeeze().detach().numpy()
                target += self.gamma * np.max(q_values)

            # Backpropagate single experience
            input_vec = np.concatenate((s, [a]))
            input_tensor = torch.tensor(input_vec, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            self.Q_sa.backpropagate(input_tensor, target_tensor)

        else:
            # Store experience in replay buffer
            self.buffer.store(s, a, r, s_next, done)

            if self.buffer.ready():
                samples= self.buffer.sample()
                if not samples:
                    return
                states, actions, rewards, next_states, dones = samples

                # Compute Q-learning targets for batch
                targets = np.copy(rewards)  # Start with current rewards

                for i in range(len(dones)):
                    if not dones[i]:  # Only update if not terminal
                        input_batch = np.stack(
                            [np.concatenate((next_states[i], [action])) for action in range(self.n_actions)],
                            axis=0
                        )
                        input_tensor = torch.tensor(input_batch, dtype=torch.float32)
                        q_values_tensor = self.query_Q(input_tensor)
                        q_values = q_values_tensor.squeeze().detach().numpy()

                        # Update target using Bellman equation
                        targets[i] += self.gamma * np.max(q_values)

                # Backpropagate for batch
                input_vecs = np.stack([np.concatenate((states[i], [actions[i]])) for i in range(len(states))], axis=0)
                input_tensor = torch.tensor(input_vecs, dtype=torch.float32)
                target_tensor = torch.tensor(targets, dtype=torch.float32)
                self.Q_sa.backpropagate(input_tensor, target_tensor)

                    


    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=500):
        returns = []  # List to store total reward for each episode
        for i in range(n_eval_episodes):
            # Reset the environment; for Gymnasium, reset returns (observation, info)
            s, _ = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                # Use the greedy policy for evaluation
                a = self.select_action(s, policy='greedy')
                # Step the environment. Gymnasium returns 5 values.
                s, r, done, truncated, _ = eval_env.step(a)
                R_ep += r
                # End episode if done or truncated
                if done or truncated:
                    break
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
