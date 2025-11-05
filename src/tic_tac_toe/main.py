from environment.ttt_env import TicTacToeEnv
from training.rl_player import TicTacToePlayer
import sys
import time
import random

# Helper methods

def prompt_int(prompt: str, default: int) -> int:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            print("Invalid number. Try again.")

def prompt_float(prompt: str, default: float) -> float:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print("Invalid number. Try again.")

def prompt_choice(prompt: str, choices: list, default: str) -> str:
    choices_str = "/".join(choices)
    while True:
        val = input(f"{prompt} ({choices_str}) [{default}]: ").strip().lower()
        if not val:
            return default
        if val in choices:
            return val
        print(f"Invalid choice. Enter one of: {choices_str}")

def prompt_bool(prompt: str, default: bool) -> bool:
    while True:
        val = input(f"{prompt} [{default}]: ").strip().lower()
        if not val:
            return default
        if val in ["y", "yes", "n", "no"]:
            return val.startswith("y")
        print("Invalid choice. Enter y/yes or n/no.")

# Training loops

def train_model(episodes: int, lr: float, epsilon: float, save_interval: int = 1000):
    print("\nStarting training...")
    player = TicTacToePlayer(lr, epsilon)
    player.load()
    env = TicTacToeEnv(player, player)

    win_counts = {1: 0, 2: 0, 0: 0}
    start_time = time.time()

    for i in range(1, episodes + 1):
        env.reset()
        episode_history = env.simulate_game()
        player.learn_from_game(episode_history)

        winner = episode_history[-1]["winner"]
        win_counts[winner] = win_counts.get(winner, 0) + 1

        # Show stats periodically
        if i % 100 == 0:
            total_games = sum(win_counts.values())
            p1_win_rate = (win_counts[1] / total_games) * 100 if total_games else 0
            draw_rate = (win_counts[0] / total_games) * 100 if total_games else 0
            p2_win_rate = (win_counts[2] / total_games) * 100 if total_games else 0
            elapsed = time.time() - start_time

            print(
                f"Episode {i}/{episodes} | "
                f"Player 1 win: {p1_win_rate:.1f}% | Draw: {draw_rate:.1f}% | Player 2 win: {p2_win_rate:.1f}% | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if i % save_interval == 0:
            player.save()

    player.save()
    print("\nTraining complete!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Final win rate (vs self): {(win_counts[1] / episodes) * 100:.1f}%\n")


def train_single_player(episodes: int, lr: float, epsilon: float, save_interval: int = 1000):
    print("\nStarting single-player training... (AI vs Player)")
    rnd = random.Random()
    player = TicTacToePlayer(lr, epsilon)
    player.load()
    env = TicTacToeEnv(player, player)
    player_starts = rnd.choice([True, False])

    win_counts = {1: 0, -1: 0, 0: 0}

    for i in range(1, episodes + 1):
        env.reset()

        # Play a full game with random opponent
        player_starts = not player_starts
        episode_history = env.run_game_one_player(player_starts)
        player.learn_from_game(episode_history)

        winner = episode_history[-1]["winner"]
        win_counts[winner] = win_counts.get(winner, 0) + 1

        if i % 10 == 0:
            total_games = sum(win_counts.values())
            p1_win_rate = (win_counts[1] / total_games) * 100
            draw_rate = (win_counts[0] / total_games) * 100
            p2_win_rate = (win_counts[2] / total_games) * 100
            print(
                f"Episode {i}/{episodes} | "
                f"Player 1 win: {p1_win_rate:.1f}% | Draw: {draw_rate:.1f}% | Player 2 win: {p2_win_rate:.1f}% | "
            )

        if i % 10 == 0:
            player.save()

    player.save()
    print("\nSingle-player training complete!")
    print(f"Final win rate (vs random): {(win_counts[1] / episodes) * 100:.1f}%\n")

# One player

def play_one_player():
    print("\nOne Player Mode — You vs AI")

    ai_player = TicTacToePlayer()
    ai_player.load()
    env = TicTacToeEnv(ai_player, ai_player)

    while True:
        env.reset()
        player_number = prompt_int("Choose player 1 or 2: ", 1)
        env.run_game_one_player(player_number == 1)
        again = prompt_bool("Play again? (y/n)", True)
        if not again:
            break

# Two player mode

def play_two_player():
    print("\nTwo Player Mode")
    env = TicTacToeEnv(None, None)
    while True:
        env.reset()
        env.run_game_two_player()
        again = prompt_bool("Play again? (y/n)", True)
        if not again:
            break

# Main

def main():
    print("XOXOXOX TIC TAC TOE XOXOXOX\n")
    print("Choose an option:\n  1. Train model\n  2. Play one player\n  3. Exit\n")

    while True:
        choice = prompt_int("Enter your choice (1-3)", 1)
        if choice == 1:
            training_type = prompt_choice("Human or AI trainer? ", ["human", "AI"], "AI")
            episodes = prompt_int("Enter number of training games", 5000)
            lr = prompt_float("Enter learning rate", 0.0001)
            epsilon = prompt_float("Enter epsilon value", 0.2)
            if training_type == "AI" or training_type == "ai":
                train_model(episodes, lr, epsilon)
            else:
                train_single_player(episodes, lr, epsilon)
        elif choice == 2:
            play_one_player()
        elif choice == 3:
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
