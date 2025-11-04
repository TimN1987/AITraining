import os
import sys
from model.single_player import RLPlayer
from training_loops.single_loop import train_single_shot


# -------------------------------
# Helper functions
# -------------------------------

def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def prompt_int(prompt: str, default: int) -> int:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            val = int(val)
            return val
        except ValueError:
            print("Invalid number. Try again.")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            val = float(val)
            return val
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
        if val in ['y', 'yes', 'n', 'no']:
            return (val == 'y') or (val == 'yes')
        print(f"Invalid choice. Enter one of: y, yes, n, no")


# -------------------------------
# Main program
# -------------------------------

def main():
    clear_console()
    print("=== BATTLESHIPS RL TRAINER ===\n")

    # Training type selection
    training_type = prompt_choice("Select training type", ["single"], default="single")

    # Default values
    default_total_games = 5000
    default_num_targets = 10
    default_lr = 1e-4
    default_epsilon = 0.2

    # Get parameters from user
    total_games = prompt_int("Total games to train", default_total_games)
    num_targets = prompt_int("Number of targets per game", default_num_targets)
    lr = prompt_float("Learning rate", default_lr)
    epsilon = prompt_float("Initial exploration rate (epsilon)", default_epsilon)
    print_stats = prompt_bool("Print detailed stats (y/n)?", False)

    # Confirm
    print("\nTraining Configuration:")
    print(f"Training type: {training_type}")
    print(f"Total games: {total_games}")
    print(f"Number of targets: {num_targets}")
    print(f"Learning rate: {lr}")
    print(f"Initial epsilon: {epsilon}\n")

    confirm = input("Proceed with training? (y/n) ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Training cancelled.")
        sys.exit(0)

    # Initialize player
    player = RLPlayer(lr=lr, epsilon=epsilon)
    player.load()

    # Start training
    if training_type == "single":
        player = train_single_shot(player, total_games=total_games, num_targets=num_targets, print_stats=print_stats)

    print("\nTraining complete. Model saved.")


if __name__ == "__main__":
    main()
