"""Experiment settings, command-line parsing, and recorded runtime metadata."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .policy import DEFAULT_CARTPOLE_SCALES, SNN_INPUT_ENCODINGS


ENVIRONMENT_ID = "CartPole-v1"
MODEL_NAMES = ("ann", "snn")


@dataclass(frozen=True)
class ExperimentConfig:
    """Every setting needed to reproduce one experiment directory."""

    environment: str
    models: list[str]
    seeds: list[int]
    total_timesteps: int
    eval_frequency: int
    periodic_eval_episodes: int
    final_eval_episodes: int
    solved_reward: float
    learning_rate: float
    rollout_steps: int
    batch_size: int
    optimisation_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coefficient: float
    value_coefficient: float
    max_gradient_norm: float
    ann_hidden_sizes: list[int]
    ann_activation: str
    snn_hidden_sizes: list[int]
    snn_input_encoding: str
    population_size: int
    population_sigma: float
    observation_scales: list[float]
    snn_steps: int
    lif_beta: float
    lif_threshold: float
    surrogate_slope: float
    inference_warmup_steps: int
    inference_timing_steps: int
    torch_threads: int
    device: str


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def runtime_metadata() -> dict[str, Any]:
    """Record the software and hardware context alongside every experiment."""

    cuda_available = torch.cuda.is_available()
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "stable_baselines3": package_version("stable-baselines3"),
        "gymnasium": package_version("gymnasium"),
        "snntorch": package_version("snntorch"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def parse_integer_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled ANN-versus-SNN PPO experiment."
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", type=parse_integer_list, default=list(range(20)))
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-frequency", type=int, default=2_500)
    parser.add_argument("--periodic-eval-episodes", type=int, default=10)
    parser.add_argument("--final-eval-episodes", type=int, default=50)
    parser.add_argument("--solved-reward", type=float, default=475.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimisation-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-gradient-norm", type=float, default=0.5)
    parser.add_argument("--ann-hidden-sizes", type=parse_integer_list, default=[64, 64])
    parser.add_argument("--snn-hidden-sizes", type=parse_integer_list, default=[59, 59])
    parser.add_argument(
        "--snn-input-encoding",
        choices=SNN_INPUT_ENCODINGS,
        default="gaussian",
        help="Use Gaussian population currents or four direct currents.",
    )
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--population-sigma", type=float, default=0.5)
    parser.add_argument(
        "--observation-scales",
        type=lambda value: [float(item.strip()) for item in value.split(",")],
        default=list(DEFAULT_CARTPOLE_SCALES),
    )
    parser.add_argument("--snn-steps", type=int, default=8)
    parser.add_argument("--lif-beta", type=float, default=0.9)
    parser.add_argument("--lif-threshold", type=float, default=0.5)
    parser.add_argument("--surrogate-slope", type=float, default=25.0)
    parser.add_argument("--inference-warmup-steps", type=int, default=100)
    parser.add_argument("--inference-timing-steps", type=int, default=5_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive_integer_fields = (
        "total_timesteps",
        "eval_frequency",
        "periodic_eval_episodes",
        "final_eval_episodes",
        "rollout_steps",
        "batch_size",
        "optimisation_epochs",
        "population_size",
        "snn_steps",
        "inference_timing_steps",
        "torch_threads",
    )
    for field in positive_integer_fields:
        if getattr(arguments, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if arguments.batch_size > arguments.rollout_steps:
        raise ValueError("--batch-size cannot exceed --rollout-steps")
    if arguments.rollout_steps % arguments.batch_size != 0:
        raise ValueError("--rollout-steps must be divisible by --batch-size")
    if len(arguments.observation_scales) != 4:
        raise ValueError("--observation-scales must contain exactly four values")
    if arguments.resume and arguments.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this PyTorch build")
