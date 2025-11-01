import os
import sys
import time
import numpy as np
from model.rl_player import RLPlayer
from environment.target_env import BattleshipsEnv

# Helper methods

def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def prompt_int(prompt: str, default: int = None, min_val: int = None, max_val: int = None) -> int:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val and default is not None:
            return default
        try:
            val = int(val)
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Please enter a number between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print("Invalid number. Try again.")

def prompt_float(prompt: str, default: float = None, min_val: float = None, max_val: float = None) -> float:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val and default is not None:
            return default
        try:
            val = float(val)
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Please enter a number between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print("Invalid number. Try again.")

def prompt_bool(prompt: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    val = input(f"{prompt} (y/n) [{default_str}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "true", "1")

def section(title: str):
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60 + "\n")

# Training loop

def train_rlplayer(player, total_games: int, num_targets: int, single_enabled: bool, airstrike_enabled: bool, bombardment_enabled: bool, save_interval: int = 100, print_stats: bool = False):
    # Number of parallel environments
    num_envs = min(8, total_games)
    envs = [BattleshipsEnv(player, num_targets, single_enabled, airstrike_enabled, bombardment_enabled, print_stats) for _ in range(num_envs)]
    episode_counts = [0 for _ in range(num_envs)]
    all_rewards = []
    all_accuracies = []
    positive_reward_total = 0
    total_steps = 0

    while min(episode_counts) < total_games // num_envs:
        for i, env in enumerate(envs):
            if episode_counts[i] >= total_games // num_envs:
                continue

            episode_history = env.run_episode()
            player.learn_from_episode(episode_history)     
            player.decay_epsilon()
            episode_counts[i] += 1
            env.reset()

            # Collect rewards for logging
            positive_reward = [step['reward'] for step in episode_history if step['reward'] > 0]
            total_reward = sum(step['reward'] for step in episode_history)
            accuracy = len(positive_reward) * 100 // len(episode_history) if len(episode_history) > 0 else -1
            all_rewards.append(total_reward)
            all_accuracies.append(accuracy)
            positive_reward_total += len(positive_reward)
            total_steps += len(episode_history)

        # Periodic logging
        avg_reward = np.mean(all_rewards[-min(200, len(all_rewards)):])
        avg_accuracy = np.mean(all_accuracies[-min(200, len(all_accuracies)):])
        print(f"Batch completed. Episodes played: {sum(episode_counts)}/{total_games}, "
              f"Avg reward: {avg_reward:.2f}, Accuracy: {avg_accuracy: .1f}%")  
        
        # Save periodically
        if sum(episode_counts) % save_interval == 0:
            print(f"Saving model at {sum(episode_counts)} episodes...")
            player.save()

    # End of game stats

    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total episodes played: {sum(episode_counts)}")
    print(f"Overall average reward: {np.mean(all_rewards):.2f}")
    print(f"Max episode reward: {np.max(all_rewards):.2f}")
    print(f"Min episode reward: {np.min(all_rewards):.2f}")
    print(f"Average accuracy: {positive_reward_total * 100 // total_steps}%")
    print("="*50 + "\n")

    player.save()
    return player

# Main program

def main():
    clear_console()
    print("BATTLESHIPS RL TRAINER\n")

    # Game configuration
    section("GAME CONFIGURATION")
    num_targets = prompt_int("How many targets?", default=10)
    single_enabled = prompt_bool("Enable single shots?", default=True)
    airstrike_enabled = prompt_bool("Enable airstrike?", default=False)
    bombardment_enabled = prompt_bool("Enable bombardment?", default=False)
    print_stats = prompt_bool("Print detailed shot stats?", default=False)

    # Training configuration
    section("TRAINING CONFIGURATION")
    total_games = prompt_int("Total games to train", default=64, min_val=1)
    epsilon = prompt_float("Exploration rate (epsilon)", default=0.2, min_val=0.0, max_val=1.0)
    lr = prompt_float("Learning rate", default=1e-4, min_val=1e-6)
    gamma = prompt_float("Discount factor (gamma)", default=0.99, min_val=0.1, max_val=0.999)

    clear_console()
    section("SUMMARY")
    print(f"Number of targets:   {num_targets}")
    print(f"Airstrike enabled:   {airstrike_enabled}")
    print(f"Bombardment enabled: {bombardment_enabled}")
    print(f"Total games:         {total_games}")
    print(f"Epsilon:             {epsilon}")
    print(f"Learning rate:       {lr}")
    print(f"Gamma:               {gamma}\n")

    confirm = input("Proceed with training? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Training cancelled.")
        sys.exit(0)

    # Initialize player
    ai_player = RLPlayer(lr=lr, epsilon=epsilon)
    ai_player.load()

    # Start training
    clear_console()
    section("TRAINING STARTED")
    start_time = time.time()

    ai_player = train_rlplayer(
        player=ai_player,
        total_games=total_games,
        num_targets=num_targets,
        single_enabled=single_enabled,
        airstrike_enabled=airstrike_enabled,
        bombardment_enabled=bombardment_enabled,
        print_stats=print_stats
    )

    elapsed = time.time() - start_time
    section("TRAINING COMPLETE")
    print(f"Training completed in {elapsed:.1f} seconds.\n")
    print("Saving model.")
    ai_player.save()


if __name__ == "__main__":
    main()
