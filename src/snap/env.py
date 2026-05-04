from agent import RLPlayer
import random

class SnapEnv:
    def __init__(self, ai_player: RLPlayer):
        self.REWARDS = {
            True: 10,
            False: -10
        }
        self.CARDS = 'ABCDEFGH'
        self.CARD_COUNT = 4

        self.player = ai_player
        self.player_hand = []

    def reset(self):
        self.player_hand.clear()
        for x in self.CARDS:
            for _ in range(self.CARD_COUNT):
                self.player_hand.append(x)
        random.shuffle(self.player_hand)

    def run_episode(self):
        self.reset()
        game_over = False
        episode_history = []
        prev = self.player_hand.pop()
        while len(self.player_hand) > 0:
            curr = self.player_hand.pop()
            state = self.player.get_state(prev, curr)
            choice = self.player.choose_action(state)
            reward = self.calculate_reward(prev, curr, choice)
            episode_history.append(
                {
                    'state': state,
                    'action': choice,
                    'reward': reward
                }
            )
            prev = curr
        return episode_history
    
    def calculate_reward(self, prev: str, curr: str, choice: bool):
        return self.REWARDS[(prev == curr) == choice]