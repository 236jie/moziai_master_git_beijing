"""
Simplified demo that mirrors RLlib's Trainer -> WorkerSet -> Rollout structure.
Run this file to see how the layers cooperate without relying on Ray internals.
"""

import random
from typing import List, Dict, Any


class RolloutWorker:
    """Single environment + policy replica that can collect experience."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.env_state = 0  # toy environment state

    def rollout(self, max_steps: int = 5) -> List[Dict[str, Any]]:
        """Mimic env interaction loop and return trajectory data."""
        trajectory = []
        for step in range(max_steps):
            obs = self.env_state
            action = self._policy(obs)
            reward, done = self._env_step(action)
            trajectory.append(
                {"worker": self.worker_id, "obs": obs, "action": action, "reward": reward}
            )
            if done:
                break
        return trajectory

    def _policy(self, obs: int) -> int:
        """Very small policy that prefers positive observations."""
        return 1 if obs >= 0 else -1

    def _env_step(self, action: int):
        """Toy environment dynamics."""
        noise = random.choice([-1, 0, 1])
        self.env_state += action + noise
        reward = -abs(self.env_state)  # reward is higher near zero
        done = abs(self.env_state) > 5
        return reward, done


class WorkerSet:
    """Container that creates and coordinates multiple RolloutWorkers."""

    def __init__(self, num_workers: int):
        self.workers = [RolloutWorker(worker_id=i) for i in range(num_workers)]

    def sample(self) -> List[Dict[str, Any]]:
        """Collect one rollout per worker and aggregate the trajectories."""
        trajectories = []
        for worker in self.workers:
            traj = worker.rollout()
            trajectories.extend(traj)
        return trajectories

    def broadcast_weights(self, weights: Any):
        """Placeholder to show how Trainer would push new policy weights."""
        print(f"[WorkerSet] broadcasting weights {weights} to {len(self.workers)} workers")


class SimpleTrainer:
    """High-level coordinator similar to RLlib's Trainer."""

    def __init__(self, num_workers: int):
        self.workers = WorkerSet(num_workers=num_workers)
        self.policy_weights = 0  # toy scalar weight

    def train(self, iterations: int = 3):
        """Main training loop: broadcast weights, sample, and update weights."""
        for it in range(iterations):
            print(f"\n[Trainer] iteration {it}")
            self.workers.broadcast_weights(self.policy_weights)
            batch = self.workers.sample()
            loss = self._compute_loss(batch)
            self._apply_gradients(loss)
            print(f"[Trainer] new weights -> {self.policy_weights}")

    def _compute_loss(self, batch: List[Dict[str, Any]]) -> float:
        """Toy loss: mean negative reward (want reward high)."""
        mean_reward = sum(item["reward"] for item in batch) / len(batch)
        loss = -mean_reward
        print(f"[Trainer] loss {loss:.3f} (mean reward {mean_reward:.3f}) from {len(batch)} steps")
        return loss

    def _apply_gradients(self, loss: float):
        """Pretend to run an optimizer step."""
        self.policy_weights -= 0.1 * loss  # gradient step


if __name__ == "__main__":
    trainer = SimpleTrainer(num_workers=2)
    trainer.train(iterations=5)


