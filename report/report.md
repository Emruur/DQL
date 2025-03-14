# Stabilizing Deep Q Learning

## Introduction

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions in an environment. It explores the environment in a trial-and-error fashion and tries to find a policy that yields the maximum cumulative reward over sequential actions.

In the simplest terms, traditional RL uses a table to store the values of each state or each state-action pair. Later, it decides on the optimal action for a particular state by choosing the action that leads to the state with the highest value. However, with large or continuous state spaces, this tabular method becomes unfeasible due to memory limitations.

In such scenarios, the tables used to store state or state-action values are approximated, often by a neural network—this approach is known as Deep Reinforcement Learning (DRL). DRL has recently gained popularity due to its successful applications in self-driving cars and game playing (e.g., AlphaGo).

However, naïve implementations of DRL that directly mimic the tabular case often result in unstable learning. This paper explores two techniques—Replay Buffers and Target Networks—to stabilize the learning of Deep Q-Learning (DQL).

## Methadology


### Q-learning

Q-learning is a model-free reinforcement learning algorithm used to find the optimal action-selection policy for a given environment. It is based on learning a Q-value function, which estimates the expected cumulative reward for taking a certain action in a given state.  

#### Key Components:
- **Q-table (Q-values, Q(s, a))**: A table storing the expected rewards for each state-action pair.  
- **Bellman Equation**: Updates Q-values iteratively based on rewards and future estimates:  

$$
  Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
$$

  - $ \alpha $ (learning rate): How much new information overrides the old.  
  - $ \gamma $ (discount factor): How much future rewards matter.  
  - $ r $ (reward): Immediate feedback from the environment.  
  - $ s' $ is the next state,
  - $ a' $ represents possible actions in the next state.

#### Learning Process:
1. Start with an empty Q-table.  
2. Select actions using an exploration-exploitation strategy (e.g., ε-greedy).  
3. Observe reward and next state.  
4. Update Q-values using the Bellman equation.  
5. Repeat until convergence.  


### Deep Q learning

In **Deep Q-Learning (DQL)**, the **Q-table** is approximated using a neural network parameterized by $ \theta $, denoted as $ Q^{\theta}(s, a)$.  

 The Q-update rule in Deep Q-Learning is given by:

$$
Q^{\theta}(s, a) \leftarrow Q^{\theta}(s, a) + \alpha \left[ r + \gamma \max_{a'} Q^{\theta}(s', a') - Q^{\theta}(s, a) \right]
$$

where:
- $ \theta $ represents the neural network parameters (weights)

However, the update cannot be performed as in the tabular case, where we can assign a specific value to the $Q(s,a)$ entry. Our Q-network has fewer free parameters than the number of state-action pairs, unlike the tabular case, which has the same number of free parameters. Therefore, some weights in the network must be shared across different $(s,a)$, meaning that modifying one $(s,a)$ entry will affect other entries. More importantly, our network is a black box, and we do not know which free parameter corresponds to a particular entry.

To guide the network, we provide it with a loss function, as done in supervised learning. This loss is computed as:

$$
r + \gamma \max_{a'} Q^{\theta}(s', a') - Q^{\theta}(s, a)
$$

where $r + \gamma \max_{a'} Q^{\theta}(s', a')$ is the value we want our network to output, and $Q^{\theta}(s, a)$ is our current prediction. The difference between these two values serves as a loss signal for backpropagation.

## Stability Problems

The naive implementation described above has several problems that can result in suboptimal training.

1. **Correlation Between Samples**  
   In the naive implementation, Q-network updates are performed on consecutive experiences, resulting in correlation between samples. This violates a key assumption in neural network training—that samples should be independent and identically distributed (i.i.d.)—and causes instability in learning.

2. **Moving Target Problem**  
   When updating $Q^{\theta}(s, a)$, we use $Q^{\theta}(s', a')$ in our target value computation. However, when an update is performed to improve $Q^{\theta}(s, a)$, the value of $Q^{\theta}(s', a')$, which was part of our target value, also gets modified. This creates a moving target problem, leading to unstable learning.

## Replay Buffers

Instead of updating the network immediately after collecting an experience, experiences from episodes can be stored in a buffer. Later, random samples from this buffer can be used to update the Q-network. This approach:

- Breaks the correlation between samples by randomizing their order.
- Allows batch training, making backpropagation more efficient.

Both of these factors increase training stability.

## Target Networks

While calculating the target value for $Q^{\theta}(s, a)$, the value $Q^{\theta}(s', a')$ is used. Note that both are obtained from the same neural network parameterized by $\theta$. This can lead to instability, as the target values keep shifting due to continuous updates to the network.

To address this issue, a separate Q-network, called the target network, is introduced. The target network, with parameters $\theta^{-}$, is a periodically updated copy of the main Q-network. Instead of using $Q^{\theta}(s', a')$, the target network provides a more stable target $Q^{\theta^-}(s', a')$, as it is not affected by updates to $Q^{\theta}(s, a)$. This results in the following update rule:

$$
Q^{\theta}(s, a) \leftarrow Q^{\theta}(s, a) + \alpha \left[ r + \gamma \max_{a'} Q^{\theta^-}(s', a') - Q^{\theta}(s, a) \right]
$$


where:

- $\theta^{-}$ represents the parameters of the target network, which are updated periodically ($\theta^{-} \gets \theta$ after a fixed number of steps).

### Combined Pseudocode
# Algorithm: Deep Q-Learning with Target Network and Replay Buffer  

**Input:** Learning rate $ \alpha $, discount factor $ \gamma $, update interval $ m $, replay buffer size $ N $, total budget.  

1. **Initialize** Q-network $ Q^{\theta}(s, a) $  
2. **Initialize** target network $ Q^{\theta^-}(s, a) \gets Q^{\theta}(s, a) $  
3. **Initialize** replay buffer $\mathcal{D}$  
4. **Sample** initial state $ s $  
5. **while** budget do:  
   - **if** at $ m^{th} $ iteration:  
     - $ Q^{\theta^-} \gets Q^{\theta} $  &nbsp;&nbsp;&nbsp;&nbsp;/* Update target network */  
   - **Sample** action $ a $ using policy (e.g., $ \epsilon $-greedy)  
   - **Simulate environment** to obtain reward and next state:  
     - $ r, s' \sim p(r, s' | s, a) $  
   - **Store experience** $(s, a, s', r)$ in replay buffer $\mathcal{D}$  
   - **Sample** $ n $ experiences from replay buffer  
   - **Perform Q-update** using sampled batch:  
     $$
     Q^{\theta}(s, a) \leftarrow Q^{\theta}(s, a) + \alpha \left[ r + \gamma \max_{a'} Q^{\theta^-}(s', a') - Q^{\theta}(s, a) \right]
     $$  
   - **Set** $ s \gets s' $  
6. **end**  

**Return:** $ Q^{\theta}(s, a) $

## Results

### Setup

#### Environments  
The Gymnasium CartPole environment has a 4D continuous state space consisting of cart position, cart velocity, pole angle, and pole angular velocity. The action space is discrete with two actions: moving the cart left (0) or right (1). The maximum number of steps is set to 150 before the environment truncates. At each timestep, the environment returns a reward of 1, meaning the longer the agent balances the pole, the more rewards it accumulates.  

#### Q Network  
The Q-network is structured as an MLP with 5 inputs and a single output, where values **< 0** indicate moving left and **> 0** indicate moving right. The number of hidden layers and their dimensions are tunable.  

The DQL algorithm is limited to 5000 steps, which was empirically shown to be sufficient for learning while remaining computationally manageable. Every 250 steps, the system is evaluated by running 30 independent episodes using the current policy and averaging the accumulated returns. The final training score is obtained by averaging the last 10 evaluations, promoting stability and early goal learning.  

### Naive DQL

The naive DQL is parameterized by the following hyperparameters. Each hyperparameter is tested with four values:

- `gamma_values` = [0.90, 0.95, 0.99, 1.0]
- `learning_rate_values` = [0.01, 0.05, 0.1, 0.2]
- `hidden_dim_values` = [8, 16, 32, 64]
- `num_hidden_layers_values` = [1, 2, 3, 4]
- `policy_values` = ['egreedy', 'softmax']
- `epsilon_values` = [0.05, 0.1, 0.2, 0.3]
- `temp_values` = [0.5, 1.0, 1.5, 2.0]

The entire configuration space is explored using grid search to find the optimal hyperparameters and assess their relative importance.

Below is the optimal configuration obtained from the grid search and the corresponding learning curve:

- **gamma:** 0.9
- **learning\_rate:** 0.01
- **hidden\_dim:** 8
- **num\_hidden\_layers:** 4
- **policy:** "softmax"
- **temp:** 2.0

![](Tunes\_1/Naive\_50.png)

To gain an insight of the parameter importance, training scores with respect to each hyperparameter are computed.

![](Tunes\_1/IndividualVariance.png)

### Replay Buffer

Introducing a replay buffer adds two hyperparameters: buffer size and batch size. The replay buffer is implemented using a deque data structure with a maximum buffer size. Batch size determines the number of experiences sampled from the buffer at each training step. Given that DQL runs for 5000 steps, the maximum buffer size considered is 5000. Smaller buffer sizes increase the sampling probability of recent experiences, equalizing their sampling chances relative to older experiences.

- `capacity_values` = [500, 1000, 2000, 5000]
- `batch_size_values` = [4, 8, 16, 32]

These hyperparameters are tested while fixing previously determined optimal hyperparameters. The optimal values found via grid search are:

- **capacity:** 5000
- **batch\_size:** 32

Below is the learning curve corresponding to this configuration:

![](Tunes\_1/RB\_50.png)

The variance attributed to each hyperparameter is computed as described previously:

![](Tunes\_1/RB\_variances.png)

### Target Network

Using a target network introduces another hyperparameter: the update interval, specifying how frequently the target network parameters are updated. The following intervals were tested:

- `target_update_values` = [10, 50, 100, 200, 500, 1000]

Again, previous hyperparameters were fixed, and only the target update interval was varied. The grid search revealed that updating the target network every 50 timesteps is optimal for this task. Below is the corresponding training curve:

![](Tunes\_1/TN\_50.png)

### Combined Results

The final training curve, using both the optimal replay buffer and target network hyperparameters determined above, is presented below:

![](Tunes\_1/RB\_TN\_50.png)


## Conclusion

## References













