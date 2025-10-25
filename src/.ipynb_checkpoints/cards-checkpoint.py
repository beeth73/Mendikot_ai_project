import random

RANKS_48 = [3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K', 'A'] # 12 ranks
SUITS = ['H', 'D', 'C', 'S']

CARD_TO_ID = {}
ID_TO_CARD = {}
card_counter = 0

for suit in SUITS:
    for rank in RANKS_48:
        card_str = f"{rank}{suit}"
        CARD_TO_ID[card_str] = card_counter
        ID_TO_CARD[card_counter] = card_str
        card_counter += 1

def get_deck_48():
    return list(range(48)) # Returns a list of integer IDs

def get_rank_suit_from_id(card_id):
    if not (0 <= card_id < 48):
        raise ValueError("Invalid card ID")
    rank_idx = card_id % 12
    suit_idx = card_id // 12
    return RANKS_48[rank_idx], SUITS[suit_idx]

def get_card_str_from_id(card_id):
    return ID_TO_CARD.get(card_id, None)

# Example Usage:
# print(CARD_TO_ID['AH'])
# print(get_card_str_from_id(11)) # AH
# print(get_rank_suit_from_id(35)) # (10, 'S')