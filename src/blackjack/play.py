import torch
import os
from agent import RLPlayer
from env import BlackJackEnv

MODEL_PATH = "bj_ai_model.pth"
PLAYER = RLPlayer()
ENV = BlackJackEnv(PLAYER)

if os.path.exists(MODEL_PATH):
    print(f"--> Found existing model '{MODEL_PATH}'. Loading weights...")
    PLAYER.policy.load_state_dict(torch.load(MODEL_PATH, map_location=PLAYER.device))
    PLAYER.epsilon = 0

def play():
    game_over = False
    ENV.set_up_game()
    print(f"The starting hand is {ENV.hand} with score {ENV.score}.")
    while not game_over:
        state = PLAYER.get_state(ENV.score / 21)
        action = PLAYER.choose_action(state)
        if action == 'twist':
            ENV.twist()
        else:
            game_over = True
        if ENV.score >= ENV.TARGET:
            game_over = True
        print(f"The AI chose to {action}, so the hand is now {ENV.hand} with score {ENV.score}.")
    if ENV.score > 21:
        print("The AI player went bust.")
    elif ENV.score == 21:
        print("BLACKJACK!!!")
    else:
        print("The AI kept it safe.")


if __name__ == "__main__":
    try:
        print("Welcome to the AI blackjack experience!")
        choice = input("What would you like to do?\n1. Play\n2. Exit")
        while choice == '1':
            games = int(input("How many games would you like to play?"))
            for _ in range(games):
                play()
            choice = input("What would you like to do?\n1. Play\n2. Exit")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")