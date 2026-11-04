"""Train and evaluate ANN and SNN PPO policies on CartPole-v1."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torch import nn

from .configuration import (
    ENVIRONMENT_ID,
    ExperimentConfig,
    parse_arguments,
    runtime_metadata,
    validate_arguments,
)
from .evaluation import (
    SeededEvaluationCallback,
    calculate_convergence,
    evaluate_seeded_episodes,
    measure_policy_latency,
)
from .policy import SNNActorCriticPolicy


def set_reproducible_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Environment and evaluation
# ---------------------------------------------------------------------------


def make_training_environment(seed: int, monitor_base_path: Path) -> Monitor:
    environment = Monitor(gym.make(ENVIRONMENT_ID), filename=str(monitor_base_path))
    environment.reset(seed=seed)
    environment.action_space.seed(seed)
    return environment


def count_trainable_parameters(model: PPO) -> int:
    return sum(
        parameter.numel()
        for parameter in model.policy.parameters()
        if parameter.requires_grad
    )


# ---------------------------------------------------------------------------
# Model construction and one complete training run
# ---------------------------------------------------------------------------


def build_model(model_name: str, environment: Monitor, seed: int, config: ExperimentConfig) -> PPO:
    common_arguments: dict[str, Any] = {
        "env": environment,
        "seed": seed,
        "verbose": 0,
        "device": config.device,
        "learning_rate": config.learning_rate,
        "n_steps": config.rollout_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.optimisation_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "ent_coef": config.entropy_coefficient,
        "vf_coef": config.value_coefficient,
        "max_grad_norm": config.max_gradient_norm,
    }

    if model_name == "ann":
        return PPO(
            policy="MlpPolicy",
            policy_kwargs={
                "activation_fn": nn.Tanh,
                "net_arch": {
                    "pi": config.ann_hidden_sizes,
                    "vf": config.ann_hidden_sizes,
                },
                "ortho_init": True,
            },
            **common_arguments,
        )
    if model_name == "snn":
        return PPO(
            policy=SNNActorCriticPolicy,
            policy_kwargs={
                "snn_hidden_sizes": config.snn_hidden_sizes,
                "input_encoding": config.snn_input_encoding,
                "population_size": config.population_size,
                "population_sigma": config.population_sigma,
                "observation_scales": config.observation_scales,
                "num_steps": config.snn_steps,
                "beta": config.lif_beta,
                "threshold": config.lif_threshold,
                "surrogate_slope": config.surrogate_slope,
                "ortho_init": True,
            },
            **common_arguments,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def run_one_model_seed(
    model_name: str,
    seed: int,
    config: ExperimentConfig,
    output_directory: Path,
) -> dict[str, Any]:
    run_name = f"{model_name}_seed{seed}"
    paths = {
        "model": output_directory / "models" / f"{run_name}.zip",
        "monitor": output_directory / "monitors" / run_name,
        "curve": output_directory / "results" / f"{run_name}_eval_curve.csv",
        "episodes": output_directory / "results" / f"{run_name}_episodes.csv",
        "summary": output_directory / "runs" / f"{run_name}_summary.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== Training {model_name.upper()} | seed {seed} | "
        f"{config.total_timesteps:,} timesteps ===",
        flush=True,
    )
    set_reproducible_state(seed)
    environment = make_training_environment(seed, paths["monitor"])
    model = build_model(model_name, environment, seed, config)
    parameter_count = count_trainable_parameters(model)
    callback = SeededEvaluationCallback(
        training_seed=seed,
        eval_frequency=config.eval_frequency,
        number_of_episodes=config.periodic_eval_episodes,
        verbose=1,
    )

    start = time.perf_counter()
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
    finally:
        environment.close()
    fit_wall_time = time.perf_counter() - start
    training_time_excluding_eval = max(
        0.0, fit_wall_time - callback.evaluation_time_seconds
    )

    model.save(paths["model"].with_suffix(""))
    final_episode_records = evaluate_seeded_episodes(
        model=model,
        episode_seed_base=300_000 + seed * 1_000,
        number_of_episodes=config.final_eval_episodes,
    )
    final_rewards = np.asarray(
        [record["reward"] for record in final_episode_records], dtype=np.float64
    )
    latency = measure_policy_latency(
        model=model,
        seed=seed,
        warmup_steps=config.inference_warmup_steps,
        timing_steps=config.inference_timing_steps,
    )
    first_convergence, sustained_convergence = calculate_convergence(
        callback.records, config.solved_reward
    )

    evaluation_curve = pd.DataFrame(callback.records)
    evaluation_curve.insert(0, "model", model_name.upper())
    evaluation_curve.to_csv(paths["curve"], index=False)

    episode_frame = pd.DataFrame(final_episode_records)
    episode_frame.insert(0, "seed", seed)
    episode_frame.insert(0, "model", model_name.upper())
    episode_frame.to_csv(paths["episodes"], index=False)

    result: dict[str, Any] = {
        "model": model_name.upper(),
        "environment": config.environment,
        "seed": seed,
        "total_timesteps": config.total_timesteps,
        "trainable_parameters": parameter_count,
        "final_mean_reward": float(final_rewards.mean()),
        "final_std_reward": float(final_rewards.std(ddof=0)),
        "final_median_reward": float(np.median(final_rewards)),
        "final_min_reward": float(final_rewards.min()),
        "final_max_reward": float(final_rewards.max()),
        "solved_episode_fraction": float(np.mean(final_rewards >= config.solved_reward)),
        "first_solved_timestep": first_convergence,
        "sustained_solved_timestep": sustained_convergence,
        "fit_wall_time_seconds": fit_wall_time,
        "periodic_evaluation_time_seconds": callback.evaluation_time_seconds,
        "training_time_excluding_eval_seconds": training_time_excluding_eval,
        **latency,
    }
    write_json(paths["summary"], result)

    print(
        f"Completed {run_name}: reward={result['final_mean_reward']:.2f} "
        f"+/- {result['final_std_reward']:.2f}, "
        f"parameters={parameter_count:,}, "
        f"training={training_time_excluding_eval:.1f}s, "
        f"latency={result['inference_latency_median_ms']:.4f}ms",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Combine independently saved runs
# ---------------------------------------------------------------------------


def combine_completed_results(output_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_records = []
    for path in sorted((output_directory / "runs").glob("*_summary.json")):
        with path.open("r", encoding="utf-8") as input_file:
            summary_records.append(json.load(input_file))
    if not summary_records:
        raise RuntimeError("No completed run summaries were found")

    summary = pd.DataFrame(summary_records).sort_values(["model", "seed"])
    summary.to_csv(output_directory / "summary.csv", index=False)

    curve_paths = sorted((output_directory / "results").glob("*_eval_curve.csv"))
    curves = pd.concat((pd.read_csv(path) for path in curve_paths), ignore_index=True)
    curves = curves.sort_values(["model", "seed", "timesteps"])
    curves.to_csv(output_directory / "eval_curve.csv", index=False)

    episode_paths = sorted((output_directory / "results").glob("*_episodes.csv"))
    episodes = pd.concat((pd.read_csv(path) for path in episode_paths), ignore_index=True)
    episodes = episodes.sort_values(["model", "seed", "episode"])
    episodes.to_csv(output_directory / "evaluation_episodes.csv", index=False)
    return summary, curves, episodes


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

    torch.set_num_threads(arguments.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = ExperimentConfig(
        environment=ENVIRONMENT_ID,
        models=list(arguments.models),
        seeds=list(arguments.seeds),
        total_timesteps=arguments.total_timesteps,
        eval_frequency=arguments.eval_frequency,
        periodic_eval_episodes=arguments.periodic_eval_episodes,
        final_eval_episodes=arguments.final_eval_episodes,
        solved_reward=arguments.solved_reward,
        learning_rate=arguments.learning_rate,
        rollout_steps=arguments.rollout_steps,
        batch_size=arguments.batch_size,
        optimisation_epochs=arguments.optimisation_epochs,
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
        clip_range=arguments.clip_range,
        entropy_coefficient=arguments.entropy_coefficient,
        value_coefficient=arguments.value_coefficient,
        max_gradient_norm=arguments.max_gradient_norm,
        ann_hidden_sizes=list(arguments.ann_hidden_sizes),
        ann_activation="Tanh",
        snn_hidden_sizes=list(arguments.snn_hidden_sizes),
        snn_input_encoding=arguments.snn_input_encoding,
        population_size=arguments.population_size,
        population_sigma=arguments.population_sigma,
        observation_scales=list(arguments.observation_scales),
        snn_steps=arguments.snn_steps,
        lif_beta=arguments.lif_beta,
        lif_threshold=arguments.lif_threshold,
        surrogate_slope=arguments.surrogate_slope,
        inference_warmup_steps=arguments.inference_warmup_steps,
        inference_timing_steps=arguments.inference_timing_steps,
        torch_threads=arguments.torch_threads,
        device=arguments.device,
    )

    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = output_directory / "experiment_config.json"
    if config_path.exists() and not (arguments.resume or arguments.overwrite):
        raise FileExistsError(
            f"{config_path} already exists; use --resume, --overwrite, or a new output directory"
        )
    if not (arguments.resume and config_path.exists()):
        write_json(
            config_path,
            {"experiment": asdict(config), "runtime": runtime_metadata()},
        )

    for model_name in config.models:
        for seed in config.seeds:
            summary_path = output_directory / "runs" / f"{model_name}_seed{seed}_summary.json"
            model_path = output_directory / "models" / f"{model_name}_seed{seed}.zip"
            if arguments.resume and summary_path.exists() and model_path.exists():
                print(f"Skipping completed run {model_name}_seed{seed}", flush=True)
                continue
            if summary_path.exists() and not arguments.overwrite:
                raise FileExistsError(
                    f"{summary_path} exists; use --resume, --overwrite, or a new output directory"
                )
            run_one_model_seed(model_name, seed, config, output_directory)

    summary, curves, episodes = combine_completed_results(output_directory)
    from .analysis import generate_analysis_artifacts

    generate_analysis_artifacts(summary, curves, output_directory)
    print(f"\nExperiment complete. Artifacts: {output_directory}", flush=True)


if __name__ == "__main__":
    main()
