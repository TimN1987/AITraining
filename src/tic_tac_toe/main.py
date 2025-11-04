from environment.ttt_env import TicTacToeEnv
from training.rl_player import TicTacToePlayer


def main():
    env = TicTacToeEnv(TicTacToePlayer(), TicTacToePlayer())
    env.run_game_two_player()

if __name__ == "__main__":
    main()