"""Matched evaluation, convergence, and policy-latency measurements."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .configuration import ENVIRONMENT_ID


def evaluate_seeded_episodes(
    model: PPO,
    episode_seed_base: int,
    number_of_episodes: int,
) -> list[dict[str, float | int]]:
    """Evaluate deterministic actions on explicitly matched episode seeds."""

    environment = gym.make(ENVIRONMENT_ID)
    records: list[dict[str, float | int]] = []
    try:
        for episode_index in range(number_of_episodes):
            episode_seed = episode_seed_base + episode_index
            observation, _ = environment.reset(seed=episode_seed)
            total_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = environment.step(
                    int(np.asarray(action).item())
                )
                total_reward += float(reward)
                episode_length += 1

            records.append(
                {
                    "episode": episode_index,
                    "episode_seed": episode_seed,
                    "reward": total_reward,
                    "length": episode_length,
                }
            )
    finally:
        environment.close()

    return records


class SeededEvaluationCallback(BaseCallback):
    """Evaluate at fixed intervals using the same episode seeds for all models."""

    def __init__(
        self,
        training_seed: int,
        eval_frequency: int,
        number_of_episodes: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.training_seed = training_seed
        self.eval_frequency = eval_frequency
        self.number_of_episodes = number_of_episodes
        self.records: list[dict[str, float | int]] = []
        self.evaluation_time_seconds = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_frequency != 0:
            return True

        start = time.perf_counter()
        episode_records = evaluate_seeded_episodes(
            model=self.model,
            episode_seed_base=100_000 + self.training_seed * 1_000,
            number_of_episodes=self.number_of_episodes,
        )
        self.evaluation_time_seconds += time.perf_counter() - start

        rewards = np.asarray(
            [record["reward"] for record in episode_records], dtype=np.float64
        )
        record: dict[str, float | int] = {
            "seed": self.training_seed,
            "timesteps": self.num_timesteps,
            "mean_eval_reward": float(rewards.mean()),
            "std_eval_reward": float(rewards.std(ddof=0)),
            "min_eval_reward": float(rewards.min()),
            "max_eval_reward": float(rewards.max()),
            "n_eval_episodes": self.number_of_episodes,
        }
        self.records.append(record)

        if self.verbose:
            print(
                f"  evaluation seed={self.training_seed} "
                f"steps={self.num_timesteps} "
                f"reward={record['mean_eval_reward']:.1f} "
                f"+/- {record['std_eval_reward']:.1f}",
                flush=True,
            )
        return True


def synchronise_device(model: PPO) -> None:
    try:
        device = next(model.policy.parameters()).device
    except StopIteration:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_policy_latency(
    model: PPO,
    seed: int,
    warmup_steps: int,
    timing_steps: int,
) -> dict[str, float | int]:
    """Measure action-selection latency without environment stepping."""

    environment = gym.make(ENVIRONMENT_ID)
    try:
        observation, _ = environment.reset(seed=200_000 + seed)
    finally:
        environment.close()

    for _ in range(warmup_steps):
        model.predict(observation, deterministic=True)
    synchronise_device(model)

    elapsed_nanoseconds = np.empty(timing_steps, dtype=np.int64)
    for index in range(timing_steps):
        synchronise_device(model)
        start = time.perf_counter_ns()
        model.predict(observation, deterministic=True)
        synchronise_device(model)
        elapsed_nanoseconds[index] = time.perf_counter_ns() - start

    elapsed_ms = elapsed_nanoseconds.astype(np.float64) / 1_000_000.0
    return {
        "inference_timing_steps": timing_steps,
        "inference_latency_mean_ms": float(elapsed_ms.mean()),
        "inference_latency_std_ms": float(elapsed_ms.std(ddof=0)),
        "inference_latency_median_ms": float(np.median(elapsed_ms)),
        "inference_latency_p95_ms": float(np.quantile(elapsed_ms, 0.95)),
    }


def calculate_convergence(
    evaluation_records: list[dict[str, float | int]],
    solved_reward: float,
) -> tuple[int | None, int | None]:
    if not evaluation_records:
        return None, None

    ordered = sorted(evaluation_records, key=lambda record: int(record["timesteps"]))
    first = next(
        (
            int(record["timesteps"])
            for record in ordered
            if float(record["mean_eval_reward"]) >= solved_reward
        ),
        None,
    )
    sustained = next(
        (
            int(record["timesteps"])
            for index, record in enumerate(ordered)
            if all(
                float(later["mean_eval_reward"]) >= solved_reward
                for later in ordered[index:]
            )
        ),
        None,
    )
    return first, sustained
