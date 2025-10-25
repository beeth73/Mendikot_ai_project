import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MendikotModel(nn.Module):
    def __init__(self, state_size, action_size, num_players=4):
        super(MendikotModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # --- THE FIX: Use the correct 512-neuron architecture ---
        self.fc_layers = nn.Sequential(
            nn.Linear(state_size, 512), 
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.ReLU()
        )
        self.policy_head = nn.Linear(512, action_size)
        self.value_head = nn.Linear(512, 1)
        # --- END OF FIX ---

    def forward(self, state_tensor):
        x = self.fc_layers(state_tensor)
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value