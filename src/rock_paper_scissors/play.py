import torch
import os
from agent import RLPlayer

MODEL_PATH = "rps_ai_model.pth"
BEATS = {
    'rock': 'paper',
    'paper': 'scissors',
    'scissors': 'rock'
}

def play(user_input):
    player = RLPlayer()
    
    if os.path.exists(MODEL_PATH) and user_input in ['rock', 'paper', 'scissors']:
        print(f"--> Found existing model '{MODEL_PATH}'. Loading weights...")
        player.policy.load_state_dict(torch.load(MODEL_PATH, map_location=player.device))
        player.epsilon = 0

        state = player.get_state(user_input)
        ai_move = player.choose_action(state)

        if ai_move == user_input:
            print(f"{ai_move} -v- {user_input} -> DRAW!!!")
        elif ai_move == BEATS[user_input]:
            print(f"{ai_move} -v- {user_input} -> AI WINS!!!")
        else:
            print(f"{ai_move} -v- {user_input} -> YOU WIN!!!")
    else:
        print("--> Game error. Cannot play.")

    

if __name__ == "__main__":
    try:
        user_input = input("Enter your selection: rock, paper, scissors: ")
        play(user_input)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")