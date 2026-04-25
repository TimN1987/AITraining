from agent import RLPlayer
import random

class RPSEnv:
    def __init__(self, ai_player: RLPlayer):
        self.rewards = {
            'draw': 1,
            'win': 5,
            'loss': -5
        }
        self.moves = ['rock', 'paper', 'scissors']
        self.beats = {
            'rock': 'scissors',
            'scissors': 'paper',
            'paper': 'rock'
        }
        self.player = ai_player
        self.random = random.Random()

    def play_game(self):
        player_move = random.choice(self.moves)
        state = self.player.get_state(player_move)
        ai_move = self.player.choose_action(state)
        reward = self.calculate_reward(player_move, ai_move)
        episode_history = {
                'state': state,
                'player move': player_move,
                'ai move': ai_move,
                'reward': reward
        }
        return episode_history
    
    def calculate_reward(self, player_move, ai_move):
        if player_move == ai_move:
            return self.rewards['draw']
        elif player_move == self.beats[ai_move]:
            return self.rewards['win']
        return self.rewards['loss']
