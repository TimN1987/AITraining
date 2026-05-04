import torch
import os
from agent import RLPlayer
from env import SnapEnv

MODEL_PATH = "snap_ai_model.pth"
PLAYER = RLPlayer()
ENV = SnapEnv(PLAYER)

if os.path.exists(MODEL_PATH):
    print(f"--> Found existing model '{MODEL_PATH}'. Loading weights...")
    PLAYER.policy.load_state_dict(torch.load(MODEL_PATH, map_location=PLAYER.device))
    PLAYER.epsilon = 0

def play():
    ENV.reset()
    print('GAME STARTING!!!')
    prev = ENV.player_hand.pop()
    while len(ENV.player_hand) > 0:
        curr = ENV.player_hand.pop()
        state = PLAYER.get_state(prev, curr)
        action = PLAYER.choose_action(state)
        print(f'The previous card was {prev}. The current card is {curr}.')
        print(f'The AI player chose {'snap' if action else 'not snap'}.')
        prev = curr
    print('GAME OVER!!!')


if __name__ == "__main__":
    try:
        print("Welcome to the AI snap experience!")
        choice = input("What would you like to do?\n1. Play\n2. Exit\n")
        while choice == '1':
            games = int(input("How many games would you like to play?\n"))
            for _ in range(games):
                play()
            choice = input("What would you like to do?\n1. Play\n2. Exit\n")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")