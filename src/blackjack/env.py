from agent import RLPlayer
import random

class BlackJackEnv:
    def __init__(self, ai_player: RLPlayer):
        self.REWARDS = {
            'blackjack': 10,
            'safe twist': 1,
            'bad twist': -1,
            'low stick': -1,
            'high stick': 1
        }
        self.TARGET = 21
        self.FACE_VALUE = 10
        self.ACE_MIN = 1
        self.ACE_MAX = 11
        self.SUITS = ['C', 'D', 'H', 'S']
        self.VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

        self.random = random.Random()
        self.player = ai_player
        self.episode_history = []
        self.game_over = False

        self.deck = []
        self.hand = []
        self.score = 0

    def set_up_game(self):
        self.episode_history.clear()
        self.reset_shuffled_deck()
        self.deal_starting_hand()
        self.game_over = False

    def reset_shuffled_deck(self):
        self.deck.clear()
        for s in self.SUITS:
            for v in self.VALUES:
                self.deck.append(v + s)
        random.shuffle(self.deck)

    def deal_starting_hand(self):
        self.hand.clear()
        for _ in range(2):
            self.hand.append(self.deck.pop())
        self.calculate_hand_value()

    def run_episode(self):
        self.set_up_game()
        while not self.game_over:
            state = self.player.get_state(self.score / 21)
            action = self.player.choose_action(state)
            if action == 'twist':
                self.twist()
            else:
                self.game_over = True
            if self.score >= self.TARGET:
                self.game_over = True
            reward = self.calculate_reward(action)
            self.episode_history.append(
                {
                    'state': state,
                    'score': self.score,
                    'ai move': action,
                    'game_over': self.game_over,
                    'reward': reward
                }
            )
        return self.episode_history

    def twist(self):
        self.hand.append(self.deck.pop())
        self.calculate_hand_value()

    def calculate_hand_value(self):
        value = 0
        contains_ace = False
        for card in self.hand:
            if card[0] == 'A':
                value += self.ACE_MIN
                contains_ace = True
            elif card[0] in ['J', 'Q', 'K']:
                value += self.FACE_VALUE
            else:
                value += int(card[:-1])
        if contains_ace and value <= 11:
            value += self.ACE_MAX - self.ACE_MIN
        self.score = value

    def calculate_reward(self, action: str) -> int:
        reward = 0
        # Rewards based on action
        if action == 'twist':
            if self.score <= self.TARGET:
                reward += self.REWARDS['safe twist']
            else:
                reward += self.REWARDS['bad twist']
        else:
            if self.score <= self.TARGET - self.FACE_VALUE:
                reward += self.REWARDS['low stick']
            else:
                reward += self.REWARDS['high stick']
        # End game rewards
        if self.game_over:
            if self.score == self.TARGET:
                reward += self.REWARDS['blackjack']
            elif self.score < self.TARGET:
                reward += self.REWARDS['blackjack'] - (self.TARGET - self.score)
            else:
                reward -= 2 * (self.score - self.TARGET)
        return reward