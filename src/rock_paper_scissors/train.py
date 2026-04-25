import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from env import RPSEnv
from agent import RLPlayer

# --- Configuration ---
MODEL_PATH = "rps_ai_model.pth"
SAVE_INTERVAL = 500  # Save every 500 episodes

def train(num_episodes, lr=0.01):
    player = RLPlayer(lr=lr)
    env = RPSEnv(player)
    
    if os.path.exists(MODEL_PATH):
        print(f"--> Found existing model '{MODEL_PATH}'. Loading weights...")
        player.policy.load_state_dict(torch.load(MODEL_PATH, map_location=player.device))
        player.epsilon = 0.05 
    else:
        print("--> No saved model found. Starting training from scratch.")

    stats = {"rewards": [], "losses": []}
    print(f"--> Training on {player.device}...")

    for ep in range(1, num_episodes + 1):
        history = env.run_episode()
        player.store_experience(history)

        loss = player.learn()

        stats["rewards"].append(history['reward'])
        if loss is not None:
            stats["losses"].append(loss)

        if ep % 100 == 0:
            avg_reward = np.mean(stats["rewards"][-100:])
            current_loss = stats["losses"][-1] if stats["losses"] else 0.0
            print(f"Ep {ep:4d} | Avg Reward: {avg_reward:5.2f} | Loss: {current_loss:.4f} | Eps: {player.epsilon:.2f}")
            player.decay_epsilon()

        if ep % SAVE_INTERVAL == 0:
            torch.save(player.policy.state_dict(), MODEL_PATH)
            print(f"--> Model checkpoint saved at episode {ep}")

    torch.save(player.policy.state_dict(), MODEL_PATH)
    print("--> Training finished. Final model saved.")
    
    plot_results(stats)

def plot_results(stats):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if len(stats["rewards"]) >= 100:
        ma = np.convolve(stats["rewards"], np.ones(100)/100, mode='valid')
        plt.plot(ma, color='red', label='100-Ep Moving Avg')
    plt.title("Reward Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(stats["losses"], color='blue', alpha=0.6)
    plt.title("Model Loss (Error)")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        user_input = input("Enter number of episodes (default 2000): ")
        num_episodes = int(user_input) if user_input.strip() else 2000
        train(num_episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")