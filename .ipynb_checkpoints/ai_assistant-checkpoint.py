# ai_assistant.py (Version 5 - Correct Trump Logic)

import torch; import numpy as np; import os; import sys; import random
if '.' not in sys.path: sys.path.insert(0, '.')
from src.agent import MendikotModel
from src.cards import CARD_TO_ID, ID_TO_CARD, get_rank_suit_from_id, RANKS_48
MODEL_PATH = 'models/2025-10-24_20-15-23/mendikot_model_ep_3000.pth'
STATE_SIZE, ACTION_SIZE = 112, 12
def state_to_tensor(hand, played_history, trump_suit, lead_suit, player_id, current_player_id, trick_cards, mendis_captured, tricks_captured):
    my_hand_vec = np.zeros(48); my_hand_vec[hand] = 1
    played_history_vec, suit_map = played_history, {'H': 0, 'D': 1, 'C': 2, 'S': 3}
    trump_vec, lead_suit_vec = np.zeros(4), np.zeros(4)
    if trump_suit is not None: trump_vec[suit_map.get(trump_suit, 0)] = 1
    if lead_suit is not None: lead_suit_vec[suit_map.get(lead_suit, 0)] = 1
    player_info = np.array([player_id/4.0, current_player_id/4.0, 0/4.0, len(trick_cards)/4.0])
    scores_vec = np.array([mendis_captured[0]/4.0, tricks_captured[0]/ACTION_SIZE, mendis_captured[1]/4.0, tricks_captured[1]/ACTION_SIZE])
    state_vector = np.concatenate([my_hand_vec, played_history_vec, trump_vec, lead_suit_vec, player_info, scores_vec])
    return torch.FloatTensor(state_vector).unsqueeze(0)
def load_trained_agent(model_path):
    print(f"Loading MendikotZero from: {model_path}");
    if not os.path.exists(model_path): print(f"FATAL ERROR: Model file not found."); return None
    model = MendikotModel(STATE_SIZE, ACTION_SIZE, 4); model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval(); print("MendikotZero is ready."); return model
def parse_card_input(input_str):
    card_ids = []
    for card in [s.strip().upper() for s in input_str.split(',')]:
        if card in CARD_TO_ID: card_ids.append(CARD_TO_ID[card])
        else: print(f"Warning: Invalid card '{card}'.")
    return card_ids
def get_legal_moves(hand, trick_cards, trick_suits_led, total_tricks_played):
    if not hand: return []
    if not trick_cards:
        if total_tricks_played < 4 and len(trick_suits_led) < 4:
            possible_leads = [c for c in hand if get_rank_suit_from_id(c)[1] not in trick_suits_led]
            return possible_leads if possible_leads else hand
        return hand
    lead_suit = get_rank_suit_from_id(trick_cards[0][1])[1]
    cards_of_lead_suit = [c for c in hand if get_rank_suit_from_id(c)[1] == lead_suit]
    return cards_of_lead_suit if cards_of_lead_suit else hand
def determine_trick_winner(trick_cards, trump_suit):
    lead_suit = get_rank_suit_from_id(trick_cards[0][1])[1]
    winning_player, lead_card = trick_cards[0]
    highest_trump, highest_lead_suit = -1, lead_card
    for p_id, c_id in trick_cards:
        _, suit = get_rank_suit_from_id(c_id)
        if suit == trump_suit:
            if c_id > highest_trump: highest_trump = c_id; winning_player = p_id
    if highest_trump != -1: return winning_player
    for p_id, c_id in trick_cards:
        _, suit = get_rank_suit_from_id(c_id)
        if suit == lead_suit:
            if c_id > highest_lead_suit: highest_lead_suit = c_id; winning_player = p_id
    return winning_player
def count_mendis_in_trick(trick_cards):
    return sum(1 for _, card_id in trick_cards if (card_id % 12) == 7)

def main():
    model = load_trained_agent(MODEL_PATH)
    if model is None: return
    print("\n--- MendikotZero AI Assistant (v5) ---")
    my_id = int(input("What is MendikotZero's seating position? (0-3): "))
    hand_str = input("Enter MendikotZero's 12 cards (e.g., AH,10S,JD,QC): ")
    my_hand = parse_card_input(hand_str); my_hand.sort()
    played_history = np.zeros(48, dtype=int)
    trump_suit, trick_suits_led = None, set()
    mendis_captured, tricks_captured = {0: 0, 1: 0}, {0: 0, 1: 0}
    current_trick_starter = 1
    while sum(tricks_captured.values()) < 12:
        print("\n" + "="*50); print(f"--- START OF TRICK #{sum(tricks_captured.values()) + 1} ---")
        current_player_id, trick_cards, lead_suit = current_trick_starter, [], None
        while len(trick_cards) < 4:
            print("-" * 20)
            print(f"Scores -> Team 0: {mendis_captured[0]}M, {tricks_captured[0]}T | Team 1: {mendis_captured[1]}M, {tricks_captured[1]}T")
            print(f"Trump: {trump_suit or 'Not Set'}")
            if current_player_id == my_id:
                print(">>> It's MendikotZero's turn to play."); print(f"Hand: {[ID_TO_CARD[c] for c in my_hand]}")
                legal_moves = get_legal_moves(my_hand, trick_cards, trick_suits_led, sum(tricks_captured.values()))
                with torch.no_grad():
                    state_tensor = state_to_tensor(my_hand, played_history, trump_suit, lead_suit, my_id, current_player_id, trick_cards, mendis_captured, tricks_captured)
                    policy_logits, _ = model(state_tensor)
                    hand_map = {card_id: i for i, card_id in enumerate(my_hand)}
                    legal_hand_indices = [hand_map[cid] for cid in legal_moves if cid in hand_map]
                    best_idx = -1; best_logit = -float('inf')
                    for idx in legal_hand_indices:
                        if policy_logits[0, idx] > best_logit: best_logit = policy_logits[0, idx]; best_idx = idx
                    chosen_card = my_hand[best_idx] if best_idx != -1 else (random.choice(legal_moves) if legal_moves else None)
                if chosen_card is None: print("AI has no cards left to play."); return
                print(f"\n==> MendikotZero wants to play: {ID_TO_CARD[chosen_card]}"); my_hand.remove(chosen_card)
                played_card_id = chosen_card
            else:
                card_str = input(f"Player {current_player_id}'s turn. What card did they play?: ")
                card_ids = parse_card_input(card_str)
                if card_ids: played_card_id = card_ids[0]
                else: print("Invalid card, please try again."); continue
            trick_cards.append((current_player_id, played_card_id))
            if trump_suit is None:
                if lead_suit is None: lead_suit = get_rank_suit_from_id(played_card_id)[1]
                else:
                    played_suit = get_rank_suit_from_id(played_card_id)[1]
                    if played_suit != lead_suit:
                        trump_suit = played_suit
                        print(f"*** Trump has been automatically set to {trump_suit}! ***")
            current_player_id = (current_player_id + 1) % 4
        print("\n--- Trick Complete ---"); print(f"Cards played: {[f'P{p}({ID_TO_CARD[c]})' for p, c in trick_cards]}")
        winner_id = determine_trick_winner(trick_cards, trump_suit)
        print(f"CALCULATED WINNER: Player {winner_id}")
        winner_team = winner_id % 2; tricks_captured[winner_team] += 1
        mendis_in_trick = count_mendis_in_trick(trick_cards)
        if mendis_in_trick > 0: print(f"DETECTED {mendis_in_trick} MENDI(S) IN THE TRICK!")
        mendis_captured[winner_team] += mendis_in_trick
        for p, c in trick_cards: played_history[c] = 1
        lead_suit = get_rank_suit_from_id(trick_cards[0][1])[1]
        trick_suits_led.add(lead_suit)
        current_trick_starter = winner_id
    print("\n--- GAME OVER ---"); print("Final Scores:")
    print(f"Team 0 (You/Gemini): {mendis_captured[0]} Mendis, {tricks_captured[0]} Tricks")
    print(f"Team 1 (MZ/Assistant): {mendis_captured[1]} Mendis, {tricks_captured[1]} Tricks")

if __name__ == "__main__":
    main()