# In src/mcts.py

import numpy as np
import math
import torch

class MCTSNode:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # A map from action to MCTSNode
        self.n_visits = 0
        self.q_value = 0.0
        self.u_value = 0.0  # UCB score
        self.prior_p = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_p=prob)

    def select(self, c_puct):
        # Select the child with the highest UCB score
        return max(self.children.items(), key=lambda act_node: act_node[1].get_ucb_score(c_puct))

    def update(self, leaf_value):
        self.n_visits += 1
        # Update Q value as the average of all evaluations in this subtree
        self.q_value += (leaf_value - self.q_value) / self.n_visits

    def get_ucb_score(self, c_puct):
        # UCB = Q(s,a) + U(s,a)
        # U = c_puct * P(s,a) * sqrt(sum of visits of parent) / (1 + visits of this node)
        if self.parent is None:
            # This case should ideally not be hit in the select logic
            # as we start selection from a node with a parent.
            # Assigning a default low value to avoid errors, but review if it's reached.
            parent_visits = 1
        else:
            parent_visits = self.parent.n_visits

        self.u_value = c_puct * self.prior_p * math.sqrt(parent_visits) / (1 + self.n_visits)
        return self.q_value + self.u_value

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, model, c_puct=1.0, n_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.root = MCTSNode()

    def _playout(self, state):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            state.play_card(state.current_player_index, action)

        # Reached a leaf node.
        # Now, get the policy and value from the neural network.
        if not state.is_game_over():
            player_id = state.current_player_index
            player_hand = state.hands[player_id]
            legal_moves = state.get_legal_moves()
            
            if not legal_moves:
                # No legal moves, this is a terminal state in a sense
                leaf_value = state.get_final_rewards()[state.get_team(player_id)]
            else:
                # Need state_to_tensor, so MCTS is better inside the Trainer
                # For now, let's assume we can get it
                state_tensor = self.state_to_tensor_func(state, player_id) # We'll pass this function in
                
                with torch.no_grad():
                    policy_logits, leaf_value_tensor = self.model(state_tensor)
                
                leaf_value = leaf_value_tensor.item()
                
                # Filter policy for legal moves
                hand_map = {card_id: i for i, card_id in enumerate(player_hand)}
                legal_hand_indices = [hand_map[card_id] for card_id in legal_moves]
                
                legal_logits = policy_logits[0, legal_hand_indices]
                action_probs = torch.softmax(legal_logits, dim=0).cpu().numpy()
                
                action_priors = {move: prob for move, prob in zip(legal_moves, action_probs)}
                node.expand(action_priors)
        else:
            # Game is over, the value is the actual reward
            player_id = state.current_player_index
            leaf_value = state.get_final_rewards()[state.get_team(player_id)]

        # Backpropagate the leaf value up the tree
        curr_node = node
        while curr_node is not None:
            curr_node.update(leaf_value)
            curr_node = curr_node.parent

    def get_move_probs(self, state, state_to_tensor_func, temp=1e-3):
        self.state_to_tensor_func = state_to_tensor_func
        for _ in range(self.n_simulations):
            state_copy = state.clone()
            self._playout(state_copy)

        # Get visit counts for actions from the root
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        
        # Create a probability distribution based on visit counts
        act_probs = torch.softmax(torch.tensor(visits, dtype=torch.float32) / temp, dim=0)
        
        return acts, act_probs