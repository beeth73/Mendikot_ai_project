from .game import GameState
from .cards import get_rank_suit_from_id, SUITS

class Player:
    def __init__(self, player_id, team_id, num_players):
        self.player_id = player_id
        self.team_id = team_id
        self.num_players = num_players
        self.hand = [] # List of card IDs

    def choose_card_to_play(self, game_state: GameState):
        raise NotImplementedError # This will be implemented by HumanPlayer and AIPlayer

    def decide_trump(self, game_state: GameState):
        raise NotImplementedError # For AI and potentially human if needed

class HumanPlayer(Player):
    def choose_card_to_play(self, game_state: GameState):
        # Display hand, prompt user for input, validate move
        pass

    def decide_trump(self, game_state: GameState):
        # Prompt user to choose trump
        pass

class AIPlayer(Player):
    def __init__(self, player_id, team_id, num_players, model):
        super().__init__(player_id, team_id, num_players)
        self.model = model # The trained neural network

    def choose_card_to_play(self, game_state: GameState):
        # Get legal moves, pass state to AI model to get best move
        pass

    def decide_trump(self, game_state: GameState):
        # Pass state to AI model to decide trump
        pass