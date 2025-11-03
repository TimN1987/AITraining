import time
import numpy as np
from model.single_player import RLPlayer
from environment.target_env_single import BattleshipsEnv


def train_single_shot(player: RLPlayer, total_games: int = 5000, num_targets: int = 10,
                      save_interval: int = 100, log_interval: int = 10, verbose: bool = True):
    """
    Trains a single-shot RLPlayer on BattleshipsEnv.
    
    Args:
        player: The RLPlayer instance to train.
        total_games: Total number of episodes to train.
        num_targets: Number of targets per episode.
        save_interval: How often to save the model (episodes).
        log_interval: How often to log stats (episodes).
        verbose: Whether to print progress.
        
    Returns:
        The trained RLPlayer.
    """
    env = BattleshipsEnv(player, num_targets, single_enabled=True,
                         airstrike_enabled=False, bombardment_enabled=False, print_stats=False)

    all_rewards = []
    all_accuracies = []
    positive_reward_total = 0
    total_steps = 0

    start_time = time.time()

    for episode in range(1, total_games + 1):
        episode_history = env.run_episode()
        player.learn_from_episode(episode_history)
        player.decay_epsilon()
        env.reset()

        # Stats
        total_reward = sum(step["reward"] for step in episode_history)
        positive_reward_count = sum(1 for step in episode_history if step["reward"] > 0)
        accuracy = (positive_reward_count * 100) // len(episode_history) if len(episode_history) > 0 else 0

        all_rewards.append(total_reward)
        all_accuracies.append(accuracy)
        positive_reward_total += positive_reward_count
        total_steps += len(episode_history)

        # Logging
        if verbose and (episode % log_interval == 0 or episode == total_games):
            avg_reward = np.mean(all_rewards[-min(100, len(all_rewards)):])
            avg_accuracy = np.mean(all_accuracies[-min(100, len(all_accuracies)):])
            print(f"Episode {episode}/{total_games} | "
                  f"Avg Reward: {avg_reward:.2f} | Accuracy: {avg_accuracy:.1f}% | "
                  f"Epsilon: {player.epsilon:.3f}")

        # Save periodically
        if episode % save_interval == 0:
            if verbose:
                print(f"Saving model at episode {episode}...")
            player.save()

    # Final summary
    if verbose:
        print("\n" + "=" * 50)
        print("SINGLE-SHOT TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total Episodes:     {total_games}")
        print(f"Average Reward:     {np.mean(all_rewards):.2f}")
        print(f"Max Episode Reward: {np.max(all_rewards):.2f}")
        print(f"Min Episode Reward: {np.min(all_rewards):.2f}")
        print(f"Average Accuracy:   {positive_reward_total * 100 // total_steps}%")
        print("=" * 50 + "\n")

    # Save final model
    player.save()

    elapsed = time.time() - start_time
    if verbose:
        print(f"Training completed in {elapsed:.1f} seconds.")

    return player
