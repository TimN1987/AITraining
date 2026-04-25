import torch
import numpy as np
import os
from env import TargetEnv
from agent import RLPlayer
import matplotlib.pyplot as plt

def train(num_episodes=5000, lr=1e-4, epsilon=0.2, device=None):
    player = RLPlayer(lr, epsilon, device)
    env = TargetEnv(player)
    
    save_interval = min(500, num_episodes)
    model_path = "target_practice_model.pth"
    
    stats = {
        "rewards": [],
        "losses": [],
        "epsilon": []
    }

    if os.path.exists(model_path):
        print(f"--> Found existing model '{model_path}'. Loading weights...")
        player.policy.load_state_dict(torch.load(model_path, map_location=player.device))
        player.epsilon = 0.1 
    else:
        print("--> No saved model found. Starting training from scratch.")

    print(f"Starting training on {player.device}...")

    for ep in range(1, num_episodes + 1):
        env.reset()

        episode_history = env.run_episode()

        state_tuple = episode_history['state']
        state_tensor, action_mask = state_tuple
        episode_history['mask'] = action_mask

        loss = player.learn(episode_history)
        reward = episode_history['reward']

        stats["rewards"].append(reward)
        if loss is not None: stats["losses"].append(loss)
        stats["epsilon"].append(player.epsilon)

        if ep % 100 == 0:
            avg_reward = np.mean(stats["rewards"][-100:])
            current_loss = loss if loss is not None else 0.0
            print(f"Ep {ep:4d} | Avg Reward: {avg_reward:6.2f} | Loss: {current_loss:.4f} | Epsilon: {player.epsilon:.2f}")
            player.decay_epsilon()

        if ep % save_interval == 0:
            torch.save(player.policy.state_dict(), model_path)
            print(f"--> Model saved at episode {ep}")

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

def safe_int(message, invalid_message="Enter a valid integer value.", min=0, max=1000000, default=0):
    print(message)
    value = input()
    if value == "":
        return default
    try:
        output = int(value)
        if output < min or output > max:
            print(f"Enter a value between {min} and {max}.")
            return safe_int(message, invalid_message, min, max, default)
        return output
    except:
        print(invalid_message)
        return safe_int(message, invalid_message, min, max, default)

def safe_float(message, invalid_message="Enter a valid decimal value.", min=0, max=1000000, default=0):
    print(message)
    value = input()
    if value == "":
        return default
    try:
        output = float(value)
        if output < min or output > max:
            print(f"Enter a value between {min} and {max}.")
            return safe_float(message, invalid_message, min, max, default)
        return output
    except:
        print(invalid_message)
        return safe_float(message, invalid_message, min, max, default)

if __name__ == "__main__":
    print("Welcome to the target practice model trainer.")
    num_episodes = safe_int("How many episodes would you like to run?", default=5000)
    lr = safe_float("Enter the learning rate.", min=1e-5, max=1e-4, default=1e-4)
    epsilon = safe_float("Enter the epsilon value for exploration.", min=0.5, max=1, default=0.2)
    train(num_episodes, lr, epsilon)