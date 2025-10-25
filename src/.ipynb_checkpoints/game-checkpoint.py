# In src/game.py

import random
import numpy as np
from .cards import get_deck_48, get_rank_suit_from_id, SUITS, CARD_TO_ID

class GameState:
    def __init__(self, num_players=4, dealer_index=0):
        self.num_players = num_players
        if num_players == 4: self.cards_per_player = 12
        else: raise ValueError("Only 4 players are supported by this version.")
        
        self.dealer_index = dealer_index
        self.current_player_index = (dealer_index + 1) % num_players
        self.deck = get_deck_48()
        self.hands = {i: [] for i in range(num_players)}
        self.trick_cards, self.trick_suits_led, self.trump_suit, self.trump_declarer_team = [], set(), None, None
        self.mendis_captured, self.tricks_captured = {0: 0, 1: 0}, {0: 0, 1: 0}
        self.mendi_suit_ids = set()
        self.current_trick_lead_player = (dealer_index + 1) % num_players
        self.current_trick_lead_suit = None
        
        # --- The attributes that were missing ---
        self.card_owner_history = np.zeros(48, dtype=int)
        self.void_suits = {p: set() for p in range(num_players)}
        
        self._initialize_mendi_ids()
        self.deal()

    def _initialize_mendi_ids(self):
        for suit_str in SUITS: self.mendi_suit_ids.add(CARD_TO_ID[f"10{suit_str}"])
    
    def deal(self):
        random.shuffle(self.deck)
        for i, card_id in enumerate(self.deck): self.hands[i % self.num_players].append(card_id)
        for hand in self.hands.values(): hand.sort()

    def get_team(self, player_idx): return player_idx % 2

    def get_legal_moves(self):
        player_hand = self.hands[self.current_player_index]
        if not player_hand: return []
        if not self.trick_cards:
            if sum(self.tricks_captured.values()) < 4 and len(self.trick_suits_led) < 4:
                possible_leads = [c for c in player_hand if get_rank_suit_from_id(c)[1] not in self.trick_suits_led]
                return possible_leads if possible_leads else player_hand
            return player_hand
        lead_suit = self.current_trick_lead_suit
        cards_of_lead_suit = [c for c in player_hand if get_rank_suit_from_id(c)[1] == lead_suit]
        return cards_of_lead_suit if cards_of_lead_suit else player_hand

    def play_card(self, player_idx, card_id):
        if player_idx != self.current_player_index: raise ValueError(f"Not Player {player_idx}'s turn.")
        if card_id not in self.hands[player_idx]: raise ValueError(f"Player {player_idx} does not have card {card_id}.")

        if self.trump_suit is None and self.current_trick_lead_suit is not None:
            _, played_suit = get_rank_suit_from_id(card_id)
            can_follow = any(get_rank_suit_from_id(c)[1] == self.current_trick_lead_suit for c in self.hands[player_idx])
            if not can_follow and played_suit != self.current_trick_lead_suit:
                self.set_trump(card_id, player_idx)
        
        self.card_owner_history[card_id] = player_idx + 1
        if self.current_trick_lead_suit is not None:
            _, played_suit = get_rank_suit_from_id(card_id)
            if played_suit != self.current_trick_lead_suit:
                self.void_suits[player_idx].add(self.current_trick_lead_suit)
        
        self.hands[player_idx].remove(card_id)
        self.trick_cards.append((player_idx, card_id))
        
        if len(self.trick_cards) == 1:
            self.current_trick_lead_player = player_idx
            _, lead_suit = get_rank_suit_from_id(card_id)
            self.current_trick_lead_suit = lead_suit
            self.trick_suits_led.add(lead_suit)
        
        if len(self.trick_cards) == self.num_players: self._resolve_trick()
        else: self.current_player_index = (self.current_player_index + 1) % self.num_players

    def _resolve_trick(self):
        lead_suit = self.current_trick_lead_suit
        winning_player, lead_card = self.trick_cards[0]
        highest_trump, highest_lead_suit = -1, lead_card
        for p_id, c_id in self.trick_cards:
            _, suit = get_rank_suit_from_id(c_id)
            if suit == self.trump_suit and c_id > highest_trump:
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
            if c_id in self.mendi_suit_ids: self.mendis_captured[winner_team] += 1
        self.current_player_index = winner_idx

    def is_game_over(self): return sum(self.tricks_captured.values()) >= self.cards_per_player

    def set_trump(self, trump_card_id, declarer_player_id):
        self.trump_suit = get_rank_suit_from_id(trump_card_id)[1]
        self.trump_declarer_team = self.get_team(declarer_player_id)

    def get_final_rewards(self):
        m0, m1, t0, t1, r = self.mendis_captured[0], self.mendis_captured[1], self.tricks_captured[0], self.tricks_captured[1], 0
        if t0 == self.cards_per_player: r = 1000
        elif t1 == self.cards_per_player: r = -1000
        elif m0 == 4: r = 600
        elif m1 == 4: r = -600
        elif m0 == 3: r = 300
        elif m1 == 3: r = -300
        elif m0 == 2:
            if t0 > t1: r = 150
            elif t1 > t0: r = -150
            else:
                if self.trump_declarer_team == 0: r = -150
                elif self.trump_declarer_team == 1: r = 150
        return {0: r, 1: -r}

    def clone(self):
        new = GameState(self.num_players, self.dealer_index)
        new.current_player_index, new.hands = self.current_player_index, {p: h[:] for p, h in self.hands.items()}
        new.trick_cards, new.trick_suits_led, new.trump_suit = self.trick_cards[:], self.trick_suits_led.copy(), self.trump_suit
        new.trump_declarer_team, new.mendis_captured, new.tricks_captured = self.trump_declarer_team, self.mendis_captured.copy(), self.tricks_captured.copy()
        new.current_trick_lead_player, new.current_trick_lead_suit = self.current_trick_lead_player, self.current_trick_lead_suit
        new.card_owner_history = self.card_owner_history.copy()
        new.void_suits = {p: s.copy() for p, s in self.void_suits.items()}
        return new