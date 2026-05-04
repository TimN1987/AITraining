from agent import RLPlayer
import random

class SnapEnv:
    def __init__(self, ai_player: RLPlayer):
        self.REWARDS = {
            'snap': 10,
            'mistake': -10
        }
        self.CARDS = 'ABCDEFGH'
        self.CARD_COUNT = 4

        self.player = ai_player
        self.player_hand = []
        self.reset()

    def reset(self):
        self.player_hand.clear()
        for x in self.CARDS:
            for _ in range(self.CARD_COUNT):
                self.player_hand.append(x)
        self.player_hand = random.shuffle(self.player_hand)