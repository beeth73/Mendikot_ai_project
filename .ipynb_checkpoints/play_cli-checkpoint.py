# play_cli.py

import torch
import numpy as np
import os
import sys
import random

# --- Setup Paths to Import Our Game Logic ---
# This ensures we can find the 'src' folder
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
if '.' not in sys.path:
    sys.path.insert(0, '.')

from src.game import GameState
from src.agent import MendikotModel
from src.cards import get_card_str_from_id

# --- Configuration ---
MODEL_PATH = 'models/2025-10-24_20-15-23/mendikot_model_ep_3000.pth' # <-- IMPORTANT: Make sure this path is correct!
STATE_SIZE = 112
ACTION_SIZE = 12

# --- Helper Functions ---
def state_to_tensor(game_state, player_id):
    # This must be the exact same function as used during training
    my_hand_vec = np.zeros(48); my_hand_vec[game_state.hands[player_id]] = 1
    played_history_vec = game_state.played_cards_history
    suit_map = {'H': 0, 'D': 1, 'C': 2, 'S': 3}
    trump_vec, lead_suit_vec = np.zeros(4), np.zeros(4)
    if game_state.trump_suit is not None: trump_vec[suit_map.get(game_state.trump_suit, 0)] = 1
    if game_state.current_trick_lead_suit is not None: lead_suit_vec[suit_map.get(game_state.current_trick_lead_suit, 0)] = 1
    player_info = np.array([player_id/4.0, game_state.current_player_index/4.0, game_state.dealer_index/4.0, len(game_state.trick_cards)/4.0])
    scores_vec = np.array([game_state.mendis_captured[0]/4.0, game_state.tricks_captured[0]/ACTION_SIZE, game_state.mendis_captured[1]/4.0, game_state.tricks_captured[1]/ACTION_SIZE])
    state_vector = np.concatenate([my_hand_vec, played_history_vec, trump_vec, lead_suit_vec, player_info, scores_vec])
    return torch.FloatTensor(state_vector).unsqueeze(0)

def load_trained_agent(model_path):
    print(f"Loading MendikotZero from: {model_path}")
    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Model file not found. Please check the MODEL_PATH variable."); return None
    model = MendikotModel(STATE_SIZE, ACTION_SIZE, 4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("MendikotZero is ready.")
    return model

# --- Player Classes ---
class Player:
    def __init__(self, name):
        self.name = name
    def choose_action(self, game, hand):
        raise NotImplementedError

class HumanPlayer(Player):
    def choose_action(self, game, hand):
        legal_moves = game.get_legal_moves()
        print("\nYour turn, Bhushan. Your hand:")
        
        # Create a map of legal card index to the card ID
        legal_move_map = {i: card_id for i, card_id in enumerate(hand) if card_id in legal_moves}

        for i, card_id in enumerate(hand):
            card_str = get_card_str_from_id(card_id)
            if card_id in legal_moves:
                print(f"  [{i}] {card_str} (Legal)")
            else:
                print(f"      {card_str} (Illegal)")

        while True:
            try:
                choice = int(input("Enter the number of the card you want to play: "))
                if choice in legal_move_map:
                    return legal_move_map[choice]
                elif choice in range(len(hand)):
                     print("That is not a legal move. You must follow suit if you can.")
                else:
                    print("Invalid number. Please choose from the numbers in brackets.")
            except ValueError:
                print("Invalid input. Please enter a number.")

class AIPlayer(Player): # This is MendikotZero
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model
    
    def choose_action(self, game, hand):
        print(f"\n{self.name}'s turn. Thinking...")
        with torch.no_grad():
            state_tensor = state_to_tensor(game, game.current_player_index)
            legal_moves = game.get_legal_moves()
            if not legal_moves: return None
            
            policy_logits, _ = self.model(state_tensor)
            hand_map = {card_id: i for i, card_id in enumerate(hand)}
            legal_hand_indices = [hand_map[card_id] for card_id in legal_moves if card_id in hand_map]
            
            best_action_idx = -1
            best_logit = -float('inf')
            for idx in legal_hand_indices:
                if policy_logits[0, idx] > best_logit:
                    best_logit = policy_logits[0, idx]
                    best_action_idx = idx
            
            return hand[best_action_idx] if best_action_idx != -1 else random.choice(legal_moves)

class SimpleLogicPlayer(Player): # This is me (Assistant/Gemini)
    def choose_action(self, game, hand):
        print(f"\n{self.name}'s turn. Deciding...")
        legal_moves = game.get_legal_moves()
        if not legal_moves: return None

        # This is a placeholder for more complex logic.
        # For now, it plays a random legal move.
        # This is the fairest and simplest "no cheating" AI.
        return random.choice(legal_moves)


# --- Main Game Loop ---
def main():
    print("--- Setting up Mendikot CLI Game ---")
    mendikot_zero_model = load_trained_agent(MODEL_PATH)
    if mendikot_zero_model is None:
        return

    players = [
        HumanPlayer("Bhushan"),
        AIPlayer("MendikotZero", mendikot_zero_model),
        SimpleLogicPlayer("Google Gemini"),
        SimpleLogicPlayer("Google Assistant")
    ]

    game = GameState(num_players=4, dealer_index=0) # Bhushan is dealer (player 0)
    
    # Announce teams and starting hands
    print("\n--- TEAMS ---")
    print("Team A: Bhushan & Google Gemini")
    print("Team B: MendikotZero & Google Assistant")

    print("\n--- THE DEAL ---")
    for i, player in enumerate(players):
        hand_str = ", ".join([get_card_str_from_id(c) for c in game.hands[i]])
        print(f"{player.name}'s Hand: {hand_str}")

    # Main game loop
    while not game.is_game_over():
        player_id = game.current_player_index
        player_obj = players[player_id]
        player_hand = game.hands[player_id]

        print("\n" + "="*30)
        print(f"Trick #{sum(game.tricks_captured.values()) + 1} | Trump: {game.trump_suit or 'Not Set'}")
        print(f"Scores: Team A (You): {game.mendis_captured[0]} Mendis, {game.tricks_captured[0]} Tricks | Team B: {game.mendis_captured[1]} Mendis, {game.tricks_captured[1]} Tricks")
        
        trick_str = "Current Trick: " + ", ".join([f"P{p_id}({get_card_str_from_id(c)})" for p_id, c in game.trick_cards])
        print(trick_str)

        chosen_card = player_obj.choose_action(game, player_hand)
        card_str = get_card_str_from_id(chosen_card)
        print(f"{player_obj.name} plays: {card_str}")

        game.play_card(player_id, chosen_card)

    print("\n" + "="*30)
    print("--- GAME OVER ---")
    final_rewards = game.get_final_rewards()
    print(f"Final Score: Team A: {game.mendis_captured[0]}M/{game.tricks_captured[0]}T | Team B: {game.mendis_captured[1]}M/{game.tricks_captured[1]}T")
    
    if final_rewards[0] > 0:
        print("Congratulations, Team A (Bhushan & Google Gemini) wins!")
    elif final_rewards[0] < 0:
        print("Team B (MendikotZero & Google Assistant) wins!")
    else:
        print("The game is a draw!")


if __name__ == "__main__":
    main()