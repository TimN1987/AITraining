import torch
import numpy as np
import os  # Required for the file check
from environment import TargetEnv
from agent import RLPlayer
import matplotlib.pyplot as plt

def train():
    # 1. Initialize
    player = RLPlayer()
    env = TargetEnv(player, airstrike_enabled=True, bombardment_enabled=True)
    
    num_episodes = 5000
    save_interval = 500
    model_path = "battleship_model.pth"
    
    stats = {
        "rewards": [],
        "losses": [],
        "epsilon": []
    }

    # --- LOADING CHECK ---
    if os.path.exists(model_path):
        print(f"--> Found existing model '{model_path}'. Loading weights...")
        player.policy.load_state_dict(torch.load(model_path, map_location=player.device))
        # Start with less randomness if we are continuing
        player.epsilon = 0.1 
    else:
        print("--> No saved model found. Starting training from scratch.")
    # ---------------------

    print(f"Starting training on {player.device}...")

    # 2. Training Loop
    for ep in range(1, num_episodes + 1):
        env.reset()

        episode_history = env.run_episode()

        # Extract data from the dictionary returned by environment
        state_tuple = episode_history['state']
        state_tensor, action_mask = state_tuple
        episode_history['mask'] = action_mask
        
        # 3. Learning Step
        loss = player.learn(episode_history)
        reward = episode_history['reward']

        # Record Stats
        stats["rewards"].append(reward)
        if loss is not None: stats["losses"].append(loss)
        stats["epsilon"].append(player.epsilon)

        # 4. Progress Reporting
        if ep % 100 == 0:
            avg_reward = np.mean(stats["rewards"][-100:])
            # Handle potential None loss for reporting
            current_loss = loss if loss is not None else 0.0
            print(f"Ep {ep:4d} | Avg Reward: {avg_reward:6.2f} | Loss: {current_loss:.4f} | Epsilon: {player.epsilon:.2f}")
            player.decay_epsilon()

        # 5. Save Model
        if ep % save_interval == 0:
            torch.save(player.policy.state_dict(), model_path)
            print(f"--> Model saved at episode {ep}")

    # 6. Final Plotting
    plot_results(stats)

def plot_results(stats):
    plt.figure(figsize=(12, 5))
    
    # Plot Rewards
    plt.subplot(1, 2, 1)
    plt.plot(stats["rewards"], alpha=0.3, color='blue', label='Raw Reward')
    # Add a moving average to see the trend clearer
    if len(stats["rewards"]) >= 100:
        ma = np.convolve(stats["rewards"], np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(stats["rewards"])), ma, color='red', label='100-Ep Average')
    plt.title("Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # Plot Losses
    plt.subplot(1, 2, 2)
    plt.plot(stats["losses"])
    plt.title("Loss over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()