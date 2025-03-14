import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CartPoleQ(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, num_hidden_layers=1, lr=1e-3):
        """
        If num_hidden_layers is 1, then the architecture is:
            fc1: input_dim -> hidden_dim
            fc2: hidden_dim -> 1
        If num_hidden_layers > 1, then:
            fc1: input_dim -> hidden_dim
            hidden_layers: a ModuleList of (num_hidden_layers - 1) layers, each hidden_dim -> hidden_dim
            fc2: hidden_dim -> 1
        """
        super(CartPoleQ, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # If more than one hidden layer, add additional layers
        if num_hidden_layers > 1:
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
            )
        else:
            # If only one hidden layer, we don't create extra hidden layers.
            self.hidden_layers = None
        
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # add batch dimension if missing
        x = F.relu(self.fc1(x))
        # Output is a raw Q-value
        q_value = self.fc2(x)
        return q_value

    def backpropagate(self, input_vector, target):
        self.optimizer.zero_grad()

        # Forward pass
        output = self.forward(input_vector)  # Expected shape: (batch_size, 1)
        
        # Ensure output matches target shape
        output = output.view(-1)  # Reshape to (batch_size,)
        target = target.view(-1)  # Converts scalar to tensor of shape (1,)
        
        # Compute loss
        loss = F.mse_loss(output, target)

        # Backward pass & optimization step
        loss.backward()
        self.optimizer.step()

        return loss.item()

# Example usage:
if __name__ == "__main__":
    q_network = CartPoleQ()
    
    # Example state: [position, velocity, angle, angular velocity]
    # and action indicator (0 for action1, 1 for action2)
    sample_state_action = torch.tensor([0.1, -0.2, 0.3, -0.1, 0], dtype=torch.float32)
    
    # Suppose we have a target Q-value from our Q-learning update
    target_value = torch.tensor(0.5, dtype=torch.float32)
    
    # Forward pass: get the predicted Q-value for the given state-action pair
    q_value = q_network.forward(sample_state_action)
    print("Predicted Q value:", q_value.item())
    
    # Perform a training (backpropagation) step
    loss = q_network.backpropagate(sample_state_action, target_value)
    print("Loss:", loss)