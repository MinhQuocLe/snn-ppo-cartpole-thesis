"""Measure final-model spike activity and estimate synaptic operations."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from .configuration import ENVIRONMENT_ID
from .policy import SpikingBranch


CONDITION_COLOURS = {
    "Gaussian SNN": "#D55E00",
    "Direct SNN": "#009E73",
}
SEED_PATTERN = re.compile(r"snn_seed(\d+)\.zip$")


@dataclass
class LayerActivityRecorder:
    """Accumulate binary spike outputs emitted by one LIF layer."""

    condition: str
    seed: int
    branch: str
    layer: int
    neurons: int
    num_steps: int
    total_spikes: float = 0.0
    total_values: int = 0
    spikes_by_neuron: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.spikes_by_neuron = np.zeros(self.neurons, dtype=np.float64)

    def __call__(self, _module, _inputs, output) -> None:
        spikes = output[0] if isinstance(output, tuple) else output
        spikes_cpu = spikes.detach().to(device="cpu", dtype=torch.float64)
        spike_count = float(spikes_cpu.sum().item())

        self.total_spikes += spike_count
        self.total_values += spikes_cpu.numel()
        self.spikes_by_neuron += spikes_cpu.sum(dim=0).numpy()

    def summary_row(self) -> dict[str, float | int | str]:
        observations = self.total_values // (self.neurons * self.num_steps)
        firing_rate = self.total_spikes / self.total_values
        return {
            "condition": self.condition,
            "seed": self.seed,
            "branch": self.branch,
            "layer": self.layer,
            "neurons": self.neurons,
            "num_steps": self.num_steps,
            "probe_observations": observations,
            "total_spikes": self.total_spikes,
            "spikes_per_observation": self.total_spikes / observations,
            "firing_rate": firing_rate,
            "spike_sparsity": 1.0 - firing_rate,
            "silent_neuron_fraction": float(np.mean(self.spikes_by_neuron == 0)),
        }


def collect_probe_observations(
    ann_model_path: Path,
    episodes: int,
    episode_seed_base: int,
) -> np.ndarray:
    """Collect realistic states from one deterministic ANN policy."""

    model = PPO.load(ann_model_path, device="cpu")
    environment = gym.make(ENVIRONMENT_ID)
    observations: list[np.ndarray] = []
    try:
        for episode_index in range(episodes):
            observation, _ = environment.reset(seed=episode_seed_base + episode_index)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = environment.step(
                    int(np.asarray(action).item())
                )
    finally:
        environment.close()

    return np.stack(observations)


def model_paths(directory: Path) -> list[tuple[int, Path]]:
    paths = []
    for path in directory.glob("snn_seed*.zip"):
        match = SEED_PATTERN.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return sorted(paths)


def attach_recorders(
    model: PPO,
    condition: str,
    seed: int,
) -> tuple[list[LayerActivityRecorder], list[torch.utils.hooks.RemovableHandle]]:
    extractor = model.policy.mlp_extractor
    branches: tuple[tuple[str, SpikingBranch], ...] = (
        ("actor", extractor.policy_net),
        ("critic", extractor.value_net),
    )
    recorders: list[LayerActivityRecorder] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for branch_name, branch in branches:
        for layer_index, lif_layer in enumerate(branch.lif_layers, start=1):
            recorder = LayerActivityRecorder(
                condition=condition,
                seed=seed,
                branch=branch_name,
                layer=layer_index,
                neurons=branch.hidden_sizes[layer_index - 1],
                num_steps=branch.num_steps,
            )
            recorders.append(recorder)
            handles.append(lif_layer.register_forward_hook(recorder))
    return recorders, handles


def measure_model_activity(
    condition: str,
    seed: int,
    path: Path,
    observations: np.ndarray,
    batch_size: int,
) -> tuple[list[dict[str, float | int | str]], PPO]:
    model = PPO.load(path, device="cpu")
    model.policy.set_training_mode(False)
    recorders, handles = attach_recorders(model, condition, seed)

    try:
        with torch.no_grad():
            for start in range(0, len(observations), batch_size):
                batch = torch.tensor(
                    observations[start : start + batch_size],
                    dtype=torch.float32,
                    device=model.device,
                )
                features = model.policy.extract_features(batch)
                if isinstance(features, tuple):
                    actor_features, critic_features = features
                else:
                    actor_features = critic_features = features
                model.policy.mlp_extractor.forward_actor(actor_features)
                model.policy.mlp_extractor.forward_critic(critic_features)
    finally:
        for handle in handles:
            handle.remove()

    return [recorder.summary_row() for recorder in recorders], model


def dense_connection_evaluations(model: PPO) -> int:
    """Count weight connections evaluated by the dense PyTorch forward pass."""

    extractor = model.policy.mlp_extractor
    total = 0
    for branch in (extractor.policy_net, extractor.value_net):
        total += branch.num_steps * sum(
            layer.in_features * layer.out_features for layer in branch.linear_layers
        )
    total += model.policy.action_net.in_features * model.policy.action_net.out_features
    total += model.policy.value_net.in_features * model.policy.value_net.out_features
    return total


def ann_connection_evaluations(model: PPO) -> int:
    """Count ANN weight connections evaluated for one action and value."""

    total = 0
    extractor = model.policy.mlp_extractor
    for branch in (extractor.policy_net, extractor.value_net):
        total += sum(
            layer.in_features * layer.out_features
            for layer in branch
            if isinstance(layer, torch.nn.Linear)
        )
    total += model.policy.action_net.in_features * model.policy.action_net.out_features
    total += model.policy.value_net.in_features * model.policy.value_net.out_features
    return total


def operation_row(
    condition: str,
    seed: int,
    model: PPO,
    activity: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Estimate operations if hidden spikes were processed event-by-event.

    Continuous inputs require multiply-accumulates (MACs). Later connections
    can use cheaper accumulates (ACs) when their presynaptic neuron spikes. The
    two types are kept in separate columns because they do not have equal
    energy cost on neuromorphic hardware.
    """

    extractor = model.policy.mlp_extractor
    continuous_input_macs = 0
    spike_triggered_accumulations = 0.0

    for branch_name, branch, output_size in (
        ("actor", extractor.policy_net, model.policy.action_net.out_features),
        ("critic", extractor.value_net, model.policy.value_net.out_features),
    ):
        first_layer = branch.linear_layers[0]
        continuous_input_macs += (
            branch.num_steps * first_layer.in_features * first_layer.out_features
        )

        branch_rows = activity[
            (activity["condition"] == condition)
            & (activity["seed"] == seed)
            & (activity["branch"] == branch_name)
        ].set_index("layer")
        for layer_index, linear_layer in enumerate(branch.linear_layers[1:], start=1):
            input_spikes = float(branch_rows.loc[layer_index, "spikes_per_observation"])
            spike_triggered_accumulations += input_spikes * linear_layer.out_features

        final_layer_number = len(branch.linear_layers)
        output_spikes = float(
            branch_rows.loc[final_layer_number, "spikes_per_observation"]
        )
        spike_triggered_accumulations += output_spikes * output_size

    return {
        "condition": condition,
        "seed": seed,
        "dense_connection_evaluations": dense_connection_evaluations(model),
        "continuous_input_macs": continuous_input_macs,
        "spike_triggered_accumulations": spike_triggered_accumulations,
        "event_driven_total_estimate": (
            continuous_input_macs + spike_triggered_accumulations
        ),
    }


def plot_activity(activity: pd.DataFrame, output_path: Path) -> None:
    labels = [
        ("actor", 1, "Actor L1"),
        ("actor", 2, "Actor L2"),
        ("critic", 1, "Critic L1"),
        ("critic", 2, "Critic L2"),
    ]
    figure, axis = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    positions = np.arange(len(labels), dtype=float)
    offsets = {"Gaussian SNN": -0.16, "Direct SNN": 0.16}
    for condition, offset in offsets.items():
        values = [
            activity[
                (activity["condition"] == condition)
                & (activity["branch"] == branch)
                & (activity["layer"] == layer)
            ]["firing_rate"].to_numpy()
            for branch, layer, _ in labels
        ]
        boxes = axis.boxplot(
            values,
            positions=positions + offset,
            widths=0.26,
            patch_artist=True,
            manage_ticks=False,
        )
        for box in boxes["boxes"]:
            box.set_facecolor(CONDITION_COLOURS[condition])
            box.set_alpha(0.72)
        for median in boxes["medians"]:
            median.set_color("black")
        axis.scatter([], [], color=CONDITION_COLOURS[condition], label=condition)

    axis.set(
        title="Final-model firing rates on a shared CartPole probe set",
        ylabel="Fraction of neurons spiking per internal step",
        xticks=positions,
        xticklabels=[label for _, _, label in labels],
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann-model",
        type=Path,
        default=Path("artifacts/cpu-20-seeds/models/ann_seed0.zip"),
    )
    parser.add_argument(
        "--gaussian-model-dir",
        type=Path,
        default=Path("artifacts/cpu-20-seeds/models"),
    )
    parser.add_argument(
        "--direct-model-dir",
        type=Path,
        default=Path("artifacts/cpu-20-seeds-direct/models"),
    )
    parser.add_argument("--gaussian-summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument(
        "--direct-summary",
        type=Path,
        default=Path("results/timestep_analysis/summary_by_seed.csv"),
    )
    parser.add_argument("--probe-episodes", type=int, default=2)
    parser.add_argument("--probe-seed-base", type=int, default=900_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("results/spike_analysis"))
    arguments = parser.parse_args()

    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    probe = collect_probe_observations(
        arguments.ann_model,
        episodes=arguments.probe_episodes,
        episode_seed_base=arguments.probe_seed_base,
    )
    pd.DataFrame(
        probe,
        columns=["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"],
    ).to_csv(output_directory / "probe_observations.csv", index=False)

    reward_tables = {
        "Gaussian SNN": pd.read_csv(arguments.gaussian_summary).query("model == 'SNN'"),
        "Direct SNN": pd.read_csv(arguments.direct_summary).query("steps == 8"),
    }
    directories = {
        "Gaussian SNN": arguments.gaussian_model_dir,
        "Direct SNN": arguments.direct_model_dir,
    }

    activity_rows: list[dict[str, float | int | str]] = []
    operation_rows: list[dict[str, float | int | str]] = []
    for condition, directory in directories.items():
        paths = model_paths(directory)
        if not paths:
            raise FileNotFoundError(f"No SNN models found in {directory}")
        model_seeds = {seed for seed, _ in paths}
        reward_seeds = set(reward_tables[condition]["seed"].astype(int))
        if model_seeds != reward_seeds:
            raise ValueError(
                f"{condition} model and reward seeds differ: "
                f"models={sorted(model_seeds)}, rewards={sorted(reward_seeds)}"
            )
        print(f"Analysing {condition}: {len(paths)} models", flush=True)
        for seed, path in paths:
            model_activity, model = measure_model_activity(
                condition=condition,
                seed=seed,
                path=path,
                observations=probe,
                batch_size=arguments.batch_size,
            )
            activity_rows.extend(model_activity)
            current_activity = pd.DataFrame(model_activity)
            operation_rows.append(
                operation_row(condition, seed, model, current_activity)
            )

    activity = pd.DataFrame(activity_rows)
    operations = pd.DataFrame(operation_rows)

    rewards = pd.concat(
        [
            frame[["seed", "final_mean_reward"]].assign(condition=condition)
            for condition, frame in reward_tables.items()
        ],
        ignore_index=True,
    )
    activity = activity.merge(rewards, on=["condition", "seed"], how="left")
    if activity["final_mean_reward"].isna().any():
        raise ValueError("At least one analysed model has no matching reward")
    ann = PPO.load(arguments.ann_model, device="cpu")
    ann_dense = ann_connection_evaluations(ann)
    operations = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "condition": "ANN",
                        "seed": 0,
                        "dense_connection_evaluations": ann_dense,
                        "continuous_input_macs": ann_dense,
                        "spike_triggered_accumulations": 0.0,
                        "event_driven_total_estimate": ann_dense,
                    }
                ]
            ),
            operations,
        ],
        ignore_index=True,
    )

    activity.to_csv(output_directory / "spike_activity_by_model.csv", index=False)
    operations.to_csv(output_directory / "operation_estimates_by_model.csv", index=False)
    operation_summary = (
        operations.groupby("condition")
        .agg(
            dense_connection_evaluations_mean=("dense_connection_evaluations", "mean"),
            continuous_input_macs_mean=("continuous_input_macs", "mean"),
            spike_triggered_accumulations_mean=(
                "spike_triggered_accumulations",
                "mean",
            ),
            event_driven_total_estimate_mean=("event_driven_total_estimate", "mean"),
            event_driven_total_estimate_std=("event_driven_total_estimate", "std"),
        )
        .reset_index()
    )
    operation_summary.to_csv(output_directory / "operation_estimates_summary.csv", index=False)

    plot_activity(activity, output_directory / "firing_rates.png")
    print(f"Spike analysis complete: {output_directory}", flush=True)


if __name__ == "__main__":
    main()
