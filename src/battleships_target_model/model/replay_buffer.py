import random
from collections import deque
from typing import List, Dict, Any

class ReplayBuffer():
    def __init__(self, capacity: int = 10000) -> None:
        """
        Simple replay buffer for storing RL transitions.
        Each transition is a dictionary containing:
            - state
            - action (position, shot_type)
            - reward
            - next_state
            - done
            - log_probs
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: dict[str, Any]) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Randomly sample a batch of transitions."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Empty the buffer."""
        self.buffer.clear()