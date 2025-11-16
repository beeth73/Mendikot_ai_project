# mendikot_agent.py
# Single-file version of your notebook suitable for GitHub Codespaces.
# Usage:
#   python mendikot_agent.py --mode train
#   python mendikot_agent.py --mode eval --checkpoint models/<run>/<file>.pth

import os
import sys
import math
import random
import argparse
from collections import deque
from itertools import product
from datetime import datetime
import joblib          # pip package name: joblib


# PyTorch and numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Optional niceties (fallback if not installed)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw):
        return x

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

# -------------------------
# Default Hyperparameters
# -------------------------
NUM_EPISODES = 50000
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SAVE_EVERY = 1000

# state/action sizes (state vector as in notebook: 272)
STATE_SIZE = 272
ACTION_SIZE = 12

# Ensure running in repo root and prepare models dir
REPO_ROOT = os.getcwd()
MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cpu")
print(f"Device: {device}")
print(f"Repo root: {REPO_ROOT}")
print(f"Models directory: {MODELS_DIR}")

# -------------------------
# Card / Game Utilities
# -------------------------
RANKS_48 = [str(r) for r in range(3, 11)] + ["J", "Q", "K", "A"]
SUITS = ["H", "D", "C", "S"]
CARD_TO_ID = {f"{rank}{suit}": i for i, (rank, suit) in enumerate(product(RANKS_48, SUITS))}
ID_TO_CARD = {i: card for card, i in CARD_TO_ID.items()}


def get_deck_48():
    return list(range(48))


def get_rank_suit_from_id(card_id):
    if not (0 <= card_id < 48):
        raise ValueError("Invalid card ID")
    rank_idx = card_id // 4
    suit_idx = card_id % 4
    return RANKS_48[rank_idx], SUITS[suit_idx]


# -------------------------
# Classes: GameState, MCTS, Model, Trainer
# -------------------------
class GameState:
    def __init__(self, num_players=4, dealer_index=0):
        self.num_players = num_players
        if num_players == 4:
            self.cards_per_player = 12
        else:
            raise ValueError("Only 4 players are supported by this version.")
        self.dealer_index = dealer_index
        self.current_player_index = (dealer_index + 1) % num_players
        self.deck = get_deck_48()
        self.hands = {i: [] for i in range(num_players)}
        # trick / scoring
        self.trick_cards = []
        self.trick_suits_led = set()
        self.trump_suit = None
        self.trump_declarer_team = None
        self.mendis_captured = {0: 0, 1: 0}
        self.tricks_captured = {0: 0, 1: 0}
        self.mendi_suit_ids = set()
        self.current_trick_lead_player = (dealer_index + 1) % num_players
        self.current_trick_lead_suit = None
        self.card_owner_history = np.zeros(48, dtype=int)
        self.void_suits = {p: set() for p in range(num_players)}
        self._initialize_mendi_ids()
        self.deal()
        # optional played history placeholder used by some helper functions
        self.played_cards_history = np.zeros(48, dtype=int)

    def _initialize_mendi_ids(self):
        for suit_str in SUITS:
            self.mendi_suit_ids.add(CARD_TO_ID[f"10{suit_str}"])

    def deal(self):
        random.shuffle(self.deck)
        for i, card_id in enumerate(self.deck):
            self.hands[i % self.num_players].append(card_id)
        for hand in self.hands.values():
            hand.sort()

    def get_team(self, player_idx):
        return player_idx % 2

    def get_legal_moves(self):
        player_hand = self.hands[self.current_player_index]
        if not player_hand:
            return []
        if not self.trick_cards:
            # lead rules: avoid repeat-suit leads when possible (as in notebook)
            if sum(self.tricks_captured.values()) < 4 and len(self.trick_suits_led) < 4:
                possible_leads = [c for c in player_hand if get_rank_suit_from_id(c)[1] not in self.trick_suits_led]
                return possible_leads if possible_leads else player_hand
            return player_hand
        lead_suit = self.current_trick_lead_suit
        cards_of_lead_suit = [c for c in player_hand if get_rank_suit_from_id(c)[1] == lead_suit]
        return cards_of_lead_suit if cards_of_lead_suit else player_hand

    def play_card(self, player_idx, card_id):
        if player_idx != self.current_player_index:
            raise ValueError(f"Not Player {player_idx}'s turn.")
        if card_id not in self.hands[player_idx]:
            raise ValueError(f"Player {player_idx} does not have card {card_id}.")

        # set trump if not set and special rule triggered
        if self.trump_suit is None:
            if self.current_trick_lead_suit is not None:
                _, played_suit = get_rank_suit_from_id(card_id)
                can_follow = any(get_rank_suit_from_id(c)[1] == self.current_trick_lead_suit for c in self.hands[player_idx])
                if not can_follow and played_suit != self.current_trick_lead_suit:
                    self.set_trump(card_id, player_idx)

        self.card_owner_history[card_id] = player_idx + 1
        # mark voids
        if self.current_trick_lead_suit is not None:
            _, played_suit = get_rank_suit_from_id(card_id)
            if played_suit != self.current_trick_lead_suit:
                self.void_suits[player_idx].add(self.current_trick_lead_suit)
        # remove and append to trick
        self.hands[player_idx].remove(card_id)
        self.trick_cards.append((player_idx, card_id))
        self.played_cards_history[card_id] = 1

        if len(self.trick_cards) == 1:
            self.current_trick_lead_player = player_idx
            _, lead_suit = get_rank_suit_from_id(card_id)
            self.current_trick_lead_suit = lead_suit
            self.trick_suits_led.add(lead_suit)

        if len(self.trick_cards) == self.num_players:
            self._resolve_trick()
        else:
            self.current_player_index = (self.current_player_index + 1) % self.num_players

    def _resolve_trick(self):
        lead_suit = self.current_trick_lead_suit
        winning_player, lead_card = self.trick_cards[0]
        highest_trump, highest_lead_suit = -1, lead_card
        for p_id, c_id in self.trick_cards:
            _, suit = get_rank_suit_from_id(c_id)
            if self.trump_suit is not None and suit == self.trump_suit and c_id > highest_trump:
                highest_trump, winning_player = c_id, p_id
        if highest_trump == -1:
            for p_id, c_id in self.trick_cards:
                _, suit = get_rank_suit_from_id(c_id)
                if suit == lead_suit and c_id > highest_lead_suit:
                    highest_lead_suit, winning_player = c_id, p_id
        self.win_trick(winning_player)
        self.trick_cards.clear()
        self.current_trick_lead_suit = None

    def win_trick(self, winner_idx):
        winner_team = self.get_team(winner_idx)
        self.tricks_captured[winner_team] += 1
        for _, c_id in self.trick_cards:
            if c_id in self.mendi_suit_ids:
                self.mendis_captured[winner_team] += 1
        self.current_player_index = winner_idx

    def is_game_over(self):
        return sum(self.tricks_captured.values()) >= self.cards_per_player

    def set_trump(self, trump_card_id, declarer_player_id):
        self.trump_suit = get_rank_suit_from_id(trump_card_id)[1]
        self.trump_declarer_team = self.get_team(declarer_player_id)

    def get_final_rewards(self):
        m0, m1 = self.mendis_captured[0], self.mendis_captured[1]
        t0, t1 = self.tricks_captured[0], self.tricks_captured[1]
        r = 0
        if t0 == self.cards_per_player:
            r = 1000
        elif t1 == self.cards_per_player:
            r = -1000
        elif m0 == 4:
            r = 600
        elif m1 == 4:
            r = -600
        elif m0 == 3:
            r = 300
        elif m1 == 3:
            r = -300
        elif m0 == 2:
            if t0 > t1:
                r = 150
            elif t1 > t0:
                r = -150
            else:
                if self.trump_declarer_team == 0:
                    r = -150
                elif self.trump_declarer_team == 1:
                    r = 150
        return {0: r, 1: -r}

    def clone(self):
        new = GameState(self.num_players, self.dealer_index)
        new.current_player_index = self.current_player_index
        new.hands = {p: h[:] for p, h in self.hands.items()}
        new.trick_cards = self.trick_cards[:]
        new.trick_suits_led = self.trick_suits_led.copy()
        new.trump_suit = self.trump_suit
        new.trump_declarer_team = self.trump_declarer_team
        new.mendis_captured = self.mendis_captured.copy()
        new.tricks_captured = self.tricks_captured.copy()
        new.current_trick_lead_player = self.current_trick_lead_player
        new.current_trick_lead_suit = self.current_trick_lead_suit
        new.card_owner_history = self.card_owner_history.copy()
        new.void_suits = {p: s.copy() for p, s in self.void_suits.items()}
        new.played_cards_history = self.played_cards_history.copy()
        return new


# -------------------------
# MCTS implementation
# -------------------------
class MCTSNode:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.q_value = 0.0
        self.prior_p = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_p=prob)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_ucb_score(c_puct))

    def update(self, leaf_value):
        self.n_visits += 1
        self.q_value += (leaf_value - self.q_value) / self.n_visits

    def get_ucb_score(self, c_puct):
        if self.parent is None:
            return self.q_value
        u_value = c_puct * self.prior_p * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.q_value + u_value

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, model, c_puct=1.0, n_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.state_to_tensor_func = None

    def _playout(self, state, node):
        # traverse
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            state.play_card(state.current_player_index, action)

        if not state.is_game_over():
            player_id = state.current_player_index
            player_hand = state.hands[player_id]
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                leaf_value = state.get_final_rewards()[state.get_team(player_id)]
            else:
                state_tensor = self.state_to_tensor_func(state, player_id)
                with torch.no_grad():
                    policy_logits, leaf_value_tensor = self.model(state_tensor)
                leaf_value = leaf_value_tensor.item()
                hand_map = {card_id: i for i, card_id in enumerate(player_hand)}
                legal_hand_indices = [hand_map[card_id] for card_id in legal_moves if card_id in hand_map]
                if not legal_hand_indices:
                    action_probs = np.ones(len(legal_moves)) / len(legal_moves)
                else:
                    legal_logits = policy_logits[0, legal_hand_indices]
                    action_probs = F.softmax(legal_logits, dim=0).cpu().numpy()
                node.expand({move: prob for move, prob in zip(legal_moves, action_probs)})
        else:
            leaf_value = state.get_final_rewards()[state.get_team(state.current_player_index)]

        curr_node = node
        while curr_node is not None:
            curr_node.update(leaf_value)
            curr_node = curr_node.parent

    def get_move_probs(self, state, state_to_tensor_func, temp=1e-3):
        self.state_to_tensor_func = state_to_tensor_func
        root = MCTSNode()
        player_id = state.current_player_index
        player_hand = state.hands[player_id]
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            return [], torch.tensor([])
        if len(legal_moves) == 1:
            return legal_moves, torch.tensor([1.0])

        state_tensor = self.state_to_tensor_func(state, player_id)
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)

        hand_map = {card_id: i for i, card_id in enumerate(player_hand)}
        legal_hand_indices = [hand_map[card_id] for card_id in legal_moves if card_id in hand_map]
        if not legal_hand_indices:
            action_probs = np.ones(len(legal_moves)) / len(legal_moves)
        else:
            legal_logits = policy_logits[0, legal_hand_indices]
            action_probs = F.softmax(legal_logits, dim=0).cpu().numpy()

        # add Dirichlet noise (self-play style)
        alpha, epsilon = 0.3, 0.25
        noise = np.random.dirichlet([alpha] * len(action_probs))
        noisy_probs = (1 - epsilon) * action_probs + epsilon * noise

        root.expand({move: prob for move, prob in zip(legal_moves, noisy_probs)})

        for _ in range(self.n_simulations):
            self._playout(state.clone(), root)

        act_visits = [(act, node.n_visits) for act, node in root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = F.softmax(torch.tensor(visits, dtype=torch.float32) / temp, dim=0)
        return acts, act_probs


# -------------------------
# Neural model
# -------------------------
class MendikotModel(nn.Module):
    def __init__(self, state_size, action_size, num_players=4):
        super(MendikotModel, self).__init__()
        self.fc_layers = nn.Sequential(nn.Linear(state_size, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.policy_head = nn.Linear(512, action_size)
        self.value_head = nn.Linear(512, 1)

    def forward(self, state_tensor):
        x = self.fc_layers(state_tensor)
        return self.policy_head(x), torch.tanh(self.value_head(x))


# -------------------------
# Replay Buffer + Trainer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, policy, reward):
        self.buffer.append((state, policy, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class MendikotTrainer:
    def __init__(self, num_players=4, state_size=STATE_SIZE, action_size=ACTION_SIZE, models_root=MODELS_DIR):
        self.num_players = num_players
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.model = MendikotModel(state_size, action_size, num_players).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_models_dir = os.path.join(models_root, run_name)
        os.makedirs(self.run_models_dir, exist_ok=True)
        print(f"Starting run {run_name}, saving to {self.run_models_dir}")

    def state_to_tensor(self, game_state, player_id):
        # Build state vector exactly as in the original notebook (272 dims)
        my_hand_vec = np.zeros(48)
        my_hand_vec[game_state.hands[player_id]] = 1
        card_owners_vec = np.zeros((48, 4))
        for card_idx in range(48):
            owner = game_state.card_owner_history[card_idx]
            if owner > 0:
                card_owners_vec[card_idx, owner - 1] = 1
        card_owners_vec = card_owners_vec.flatten()

        void_vec = np.zeros((4, 4))
        suit_map = {"H": 0, "D": 1, "C": 2, "S": 3}
        for p_id in range(4):
            for suit_char in game_state.void_suits[p_id]:
                void_vec[p_id, suit_map[suit_char]] = 1
        void_vec = void_vec.flatten()

        trump_vec = np.zeros(4)
        lead_suit_vec = np.zeros(4)
        if game_state.trump_suit is not None:
            trump_vec[suit_map.get(game_state.trump_suit, 0)] = 1
        if game_state.current_trick_lead_suit is not None:
            lead_suit_vec[suit_map.get(game_state.current_trick_lead_suit, 0)] = 1

        player_info = np.array([player_id / 4.0, game_state.current_player_index / 4.0, game_state.dealer_index / 4.0,
                                len(game_state.trick_cards) / 4.0])
        scores_vec = np.array([game_state.mendis_captured[0] / 4.0, game_state.tricks_captured[0] / self.action_size,
                               game_state.mendis_captured[1] / 4.0, game_state.tricks_captured[1] / self.action_size])

        state_vector = np.concatenate([my_hand_vec, card_owners_vec, void_vec, trump_vec, lead_suit_vec, player_info, scores_vec])
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

    def choose_action(self, game_state, temp=1.0):
        mcts = MCTS(self.model, n_simulations=100)
        legal_moves, move_probs = mcts.get_move_probs(game_state, self.state_to_tensor, temp=temp)
        if not legal_moves:
            return None, None
        chosen_idx = np.random.choice(len(legal_moves), p=move_probs.numpy())
        chosen_card = legal_moves[chosen_idx]
        # map probabilities into action-size vector for learning
        move_probs_for_learning = torch.zeros(self.action_size, device=self.device)
        hand_map = {card_id: i for i, card_id in enumerate(game_state.hands[game_state.current_player_index])}
        for move, prob in zip(legal_moves, move_probs):
            if move in hand_map:
                move_probs_for_learning[hand_map[move]] = prob
        return chosen_card, move_probs_for_learning

    def run_episode(self):
        game = GameState(num_players=self.num_players)
        episode_history = []
        while not game.is_game_over():
            player_id = game.current_player_index
            action_card, move_probs = self.choose_action(game)
            if action_card is None:
                break
            state_tensor = self.state_to_tensor(game, player_id)
            episode_history.append({"state": state_tensor, "policy": move_probs, "player": player_id})
            game.play_card(player_id, action_card)
        final_rewards = game.get_final_rewards()
        for step in episode_history:
            team_id = game.get_team(step["player"])
            step_reward = final_rewards[team_id]
            self.replay_buffer.push(step["state"], step["policy"], step_reward)

    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, target_policies, rewards = zip(*batch)
        states_tensor = torch.cat(states).to(self.device)
        target_policies_tensor = torch.stack(target_policies).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        pred_policies_logits, pred_values = self.model(states_tensor)
        pred_values = pred_values.squeeze()

        value_loss = F.mse_loss(pred_values, rewards_tensor)
        # CrossEntropy expects class indices; to keep the same semantics as the notebook
        # we'll use KL divergence / MSE on probabilities to avoid shape mismatch.
        policy_loss = F.mse_loss(F.softmax(pred_policies_logits, dim=1), target_policies_tensor)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train(self, n_episodes=NUM_EPISODES, save_every=SAVE_EVERY):
        for episode in tqdm(range(n_episodes)):
            self.run_episode()
            self.learn()
            if (episode + 1) % save_every == 0:
                save_path = os.path.join(self.run_models_dir, f"mendikot_model_ep_{episode + 1}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved model to {save_path}")
                try:
                    joblib_path = os.path.join(self.run_models_dir, f"mendikot_model_ep_{episode + 1}.joblib")
                    # ensure the state_dict tensors are on CPU and converted to numpy if necessary
                    state_dict_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in self.model.state_dict().items()}
                    # joblib can serialize PyTorch tensors; we keep tensors (on CPU) which is fine
                    joblib.dump(state_dict_cpu, joblib_path)
                    print(f"Saved joblib checkpoint to {joblib_path}")
                except Exception as e:
                    print(f"Warning: failed to save joblib checkpoint: {e}")
        print("Training complete!")


# -------------------------
# Evaluation utilities
# -------------------------
class RandomAgent:
    def choose_action(self, game_state):
        legal_moves = game_state.get_legal_moves()
        return int(np.random.choice(legal_moves)) if legal_moves else None


def state_to_tensor_for_eval(game_state, player_id):
    my_hand_vec = np.zeros(48)
    my_hand_vec[game_state.hands[player_id]] = 1
    played_history_vec = game_state.played_cards_history
    suit_map = {"H": 0, "D": 1, "C": 2, "S": 3}
    trump_vec = np.zeros(4)
    lead_suit_vec = np.zeros(4)
    if game_state.trump_suit is not None:
        trump_vec[suit_map.get(game_state.trump_suit, 0)] = 1
    if game_state.current_trick_lead_suit is not None:
        lead_suit_vec[suit_map.get(game_state.current_trick_lead_suit, 0)] = 1
    player_info = np.array([player_id / 4.0, game_state.current_player_index / 4.0, game_state.dealer_index / 4.0,
                            len(game_state.trick_cards) / 4.0])
    scores_vec = np.array([game_state.mendis_captured[0] / 4.0, game_state.tricks_captured[0] / ACTION_SIZE,
                           game_state.mendis_captured[1] / 4.0, game_state.tricks_captured[1] / ACTION_SIZE])
    state_vector = np.concatenate([my_hand_vec, played_history_vec, trump_vec, lead_suit_vec, player_info, scores_vec])
    return torch.FloatTensor(state_vector).unsqueeze(0)


def load_trained_agent(model_path, state_size=STATE_SIZE, action_size=ACTION_SIZE, num_players=4):
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: checkpoint not found: {model_path}")
        return None
    model = MendikotModel(state_size, action_size, num_players)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, num_games=500, ai_player_id=0):
    wins = 0
    total_rewards = []
    random_agent = RandomAgent()
    for _ in tqdm(range(num_games), desc="Evaluating Model"):
        game = GameState(num_players=4)
        ai_team = game.get_team(ai_player_id)
        while not game.is_game_over():
            player_id = game.current_player_index
            chosen_card = None
            if game.get_team(player_id) == ai_team:
                with torch.no_grad():
                    state_tensor = state_to_tensor_for_eval(game, player_id).to(device)
                    player_hand = game.hands[player_id]
                    legal_moves = game.get_legal_moves()
                    if not legal_moves:
                        break
                    # get best action from policy head over hand indices
                    policy_logits, _ = model(state_tensor)
                    hand_map = {card_id: i for i, card_id in enumerate(player_hand)}
                    legal_hand_indices = [hand_map[card_id] for card_id in legal_moves if card_id in hand_map]
                    best_action_idx = -1
                    best_logit = -float("inf")
                    for idx in legal_hand_indices:
                        if policy_logits[0, idx] > best_logit:
                            best_logit = policy_logits[0, idx]
                            best_action_idx = idx
                    if best_action_idx != -1 and best_action_idx < len(player_hand):
                        chosen_card = player_hand[best_action_idx]
            else:
                chosen_card = random_agent.choose_action(game)
            if chosen_card is None:
                break
            game.play_card(player_id, chosen_card)
        ai_reward = game.get_final_rewards()[ai_team]
        total_rewards.append(ai_reward)
        if ai_reward > 0:
            wins += 1
    return wins, total_rewards


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Mendikot training/evaluation runner")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="train or eval")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="number of training episodes")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to model checkpoint for eval")
    parser.add_argument("--eval-games", type=int, default=500, help="number of games to simulate in eval")
    args = parser.parse_args()

    if args.mode == "train":
        # keep a conservative default for codespaces to avoid long runs accidentally
        n_eps = args.episodes
        print("Training mode - be careful: this can be long in Codespaces.")
        trainer = MendikotTrainer(num_players=4, state_size=STATE_SIZE, action_size=ACTION_SIZE, models_root=MODELS_DIR)
        trainer.train(n_episodes=n_eps, save_every=SAVE_EVERY)

    elif args.mode == "eval":
        if not args.checkpoint:
            print("Please provide --checkpoint models/<run>/<file>.pth for evaluation.")
            return
        model = load_trained_agent(args.checkpoint, STATE_SIZE, ACTION_SIZE, 4)
        if model is None:
            return
        wins, rewards = evaluate_model(model, num_games=args.eval_games, ai_player_id=0)
        win_rate = (wins / args.eval_games) * 100
        avg_reward = float(np.mean(rewards))
        print("\n--- Evaluation Results ---")
        print(f"Model: {args.checkpoint}")
        print(f"Games Played: {args.eval_games}")
        print(f"Win Rate vs Random Agents: {win_rate:.2f}%")
        print(f"Average Reward per Game: {avg_reward:.2f}")

        if HAS_PLOTTING:
            plt.style.use("seaborn-whitegrid")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(rewards, bins=20, kde=True, ax=ax)
            ax.set_title(f"Reward Distribution - {os.path.basename(args.checkpoint)}")
            ax.set_xlabel("Final Reward")
            ax.set_ylabel("Number of Games")
            ax.axvline(avg_reward, color="red", linestyle="--", label=f"Avg: {avg_reward:.2f}")
            ax.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("(matplotlib / seaborn not available; skipping plot.)")


if __name__ == "__main__":
    main()
