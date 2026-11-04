"""Compare direct-input SNNs using 4, 8, and 16 internal steps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import MAXIMUM_REWARD, SOLVED_REWARD, configure_plot_style
from .spike_analysis import measure_model_activity, model_paths, operation_row


STEP_COLOURS = {4: "#56B4E9", 8: "#009E73", 16: "#CC79A7"}
CONTRASTS = ((4, 8), (8, 16), (4, 16))


def exact_mcnemar_p_value(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""

    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    lower_tail = sum(
        math.comb(discordant, successes) for successes in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def read_configuration(directory: Path) -> dict[str, Any]:
    with (directory / "experiment_config.json").open(encoding="utf-8") as file:
        return json.load(file)["experiment"]


def load_conditions(
    directories: dict[int, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the matched timestep conditions."""

    summaries: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    reference_configuration: dict[str, Any] | None = None
    reference_seeds: set[int] | None = None

    for steps, directory in directories.items():
        configuration = read_configuration(directory)
        if configuration["snn_steps"] != steps:
            raise ValueError(
                f"Expected {steps} steps in {directory}, found "
                f"{configuration['snn_steps']}"
            )
        if configuration["models"] != ["snn"]:
            raise ValueError(f"{directory} is not an SNN-only condition")

        controlled_configuration = dict(configuration)
        controlled_configuration.pop("snn_steps")
        if reference_configuration is None:
            reference_configuration = controlled_configuration
        elif controlled_configuration != reference_configuration:
            differing_fields = sorted(
                field
                for field in set(reference_configuration) | set(controlled_configuration)
                if reference_configuration.get(field) != controlled_configuration.get(field)
            )
            raise ValueError(
                "Timestep conditions differ in controlled fields: "
                + ", ".join(differing_fields)
            )

        summary = pd.read_csv(directory / "summary.csv")
        curve = pd.read_csv(directory / "eval_curve.csv")
        seeds = set(summary["seed"].astype(int))
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
            raise ValueError(
                f"Training seeds do not match for {steps} steps: {sorted(seeds)}"
            )
        saved_model_seeds = {seed for seed, _ in model_paths(directory / "models")}
        if saved_model_seeds != seeds:
            raise ValueError(
                f"Saved model seeds do not match summary seeds for {steps} steps"
            )

        summary.insert(0, "steps", steps)
        curve.insert(0, "steps", steps)
        summaries.append(summary)
        curves.append(curve)

    return (
        pd.concat(summaries, ignore_index=True).sort_values(["steps", "seed"]),
        pd.concat(curves, ignore_index=True).sort_values(
            ["steps", "seed", "timesteps"]
        ),
    )


def aggregate_conditions(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for steps, rows_for_steps in summary.groupby("steps", sort=True):
        final_reward = rows_for_steps["final_mean_reward"]
        first_solved = rows_for_steps["first_solved_timestep"].dropna()
        sustained = rows_for_steps["sustained_solved_timestep"].dropna()
        # A sustained timestamp exists exactly when the final periodic checkpoint
        # is above threshold: the last checkpoint is itself a valid one-point
        # suffix. Report that count plainly instead of implying that a policy
        # stayed above threshold continuously after its first crossing.
        rows.append(
            {
                "steps": int(steps),
                "runs": len(rows_for_steps),
                "reward_mean": final_reward.mean(),
                "reward_std": final_reward.std(ddof=1),
                "reward_median": final_reward.median(),
                "reward_min": final_reward.min(),
                "reward_max": final_reward.max(),
                "successful_runs": int((final_reward >= SOLVED_REWARD).sum()),
                "perfect_runs": int((final_reward == MAXIMUM_REWARD).sum()),
                "first_threshold_count": len(first_solved),
                "first_threshold_median": first_solved.median(),
                "final_periodic_success_count": len(sustained),
                "final_suffix_start_median": sustained.median(),
                "training_time_median_seconds": rows_for_steps[
                    "training_time_excluding_eval_seconds"
                ].median(),
                "latency_median_ms": rows_for_steps[
                    "inference_latency_median_ms"
                ].median(),
                "parameters": int(rows_for_steps["trainable_parameters"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def paired_reward_analysis(
    summary: pd.DataFrame,
    bootstrap_samples: int = 20_000,
    random_seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = summary.pivot(index="seed", columns="steps", values="final_mean_reward")
    random_generator = np.random.default_rng(random_seed)
    difference_rows = []
    comparison_rows = []
    for lower_steps, upper_steps in CONTRASTS:
        matched = pivot[[lower_steps, upper_steps]].dropna()
        differences = (
            matched[upper_steps] - matched[lower_steps]
        ).to_numpy(dtype=np.float64)
        for seed, difference in zip(matched.index, differences, strict=True):
            difference_rows.append(
                {
                    "contrast": f"{upper_steps} minus {lower_steps}",
                    "seed": int(seed),
                    "lower_steps": lower_steps,
                    "upper_steps": upper_steps,
                    "reward_difference": float(difference),
                }
            )
        indices = random_generator.integers(
            0, len(differences), size=(bootstrap_samples, len(differences))
        )
        bootstrapped_means = differences[indices].mean(axis=1)
        lower_success = matched[lower_steps] >= SOLVED_REWARD
        upper_success = matched[upper_steps] >= SOLVED_REWARD
        upper_only = int((upper_success & ~lower_success).sum())
        lower_only = int((lower_success & ~upper_success).sum())
        comparison_rows.append(
            {
                "contrast": f"{upper_steps} minus {lower_steps}",
                "paired_seeds": len(differences),
                "mean_reward_difference": differences.mean(),
                "median_reward_difference": np.median(differences),
                "bootstrap_95_ci_low": np.quantile(bootstrapped_means, 0.025),
                "bootstrap_95_ci_high": np.quantile(bootstrapped_means, 0.975),
                "success_upper_only": upper_only,
                "success_lower_only": lower_only,
                "mcnemar_exact_p": exact_mcnemar_p_value(upper_only, lower_only),
            }
        )
    return pd.DataFrame(difference_rows), pd.DataFrame(comparison_rows)


def analyse_spikes(
    directories: dict[int, Path],
    probe: np.ndarray,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    activity_rows: list[dict[str, float | int | str]] = []
    operation_rows: list[dict[str, float | int | str]] = []

    for steps, directory in directories.items():
        condition = f"{steps} steps"
        paths = model_paths(directory / "models")
        print(f"Analysing spike activity: {condition}, {len(paths)} models", flush=True)
        for seed, model_path in paths:
            model_activity, model = measure_model_activity(
                condition=condition,
                seed=seed,
                path=model_path,
                observations=probe,
                batch_size=batch_size,
            )
            for row in model_activity:
                row["steps"] = steps
            activity_rows.extend(model_activity)
            operation = operation_row(
                condition,
                seed,
                model,
                pd.DataFrame(model_activity),
            )
            operation["steps"] = steps
            operation_rows.append(operation)

    return pd.DataFrame(activity_rows), pd.DataFrame(operation_rows)


def plot_learning_curves(curves: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
    for steps in (4, 8, 16):
        rows = curves[curves["steps"] == steps]
        aggregate = rows.groupby("timesteps")["mean_eval_reward"].agg(["mean", "std"])
        x = aggregate.index.to_numpy(dtype=float)
        mean = aggregate["mean"].to_numpy(dtype=float)
        deviation = aggregate["std"].to_numpy(dtype=float)
        axis.plot(x, mean, color=STEP_COLOURS[steps], linewidth=2, label=f"{steps} steps")
        axis.fill_between(
            x,
            np.clip(mean - deviation, 0, MAXIMUM_REWARD),
            np.clip(mean + deviation, 0, MAXIMUM_REWARD),
            color=STEP_COLOURS[steps],
            alpha=0.14,
        )
    axis.axhline(SOLVED_REWARD, color="#555555", linestyle="--", linewidth=1)
    axis.set(
        title="Learning performance across twenty matched seeds",
        xlabel="Environment interactions",
        ylabel="Mean periodic evaluation reward",
        ylim=(0, 510),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.savefig(output_path)
    plt.close(figure)


def plot_computational_cost(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.1), constrained_layout=True)
    for axis, metric, title, ylabel in (
        (
            axes[0],
            "training_time_excluding_eval_seconds",
            "PPO training time",
            "Seconds",
        ),
        (
            axes[1],
            "inference_latency_median_ms",
            "Action-selection latency",
            "Milliseconds",
        ),
    ):
        values = [
            summary.loc[summary["steps"] == steps, metric].to_numpy(dtype=float)
            for steps in (4, 8, 16)
        ]
        boxes = axis.boxplot(values, tick_labels=(4, 8, 16), patch_artist=True)
        for box, steps in zip(boxes["boxes"], (4, 8, 16), strict=True):
            box.set_facecolor(STEP_COLOURS[steps])
            box.set_alpha(0.75)
        axis.set(title=title, xlabel="Internal steps", ylabel=ylabel)
        if metric == "training_time_excluding_eval_seconds":
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("CPU cost increases with internal simulation steps")
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps4-dir",
        type=Path,
        default=Path("artifacts/cpu-20-seeds-direct-steps4"),
    )
    parser.add_argument(
        "--steps8-dir",
        type=Path,
        default=Path("artifacts/cpu-20-seeds-direct"),
    )
    parser.add_argument(
        "--steps16-dir",
        type=Path,
        default=Path("artifacts/cpu-20-seeds-direct-steps16"),
    )
    parser.add_argument(
        "--probe-observations",
        type=Path,
        default=Path("results/spike_analysis/probe_observations.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/timestep_analysis")
    )
    arguments = parser.parse_args()

    directories = {
        4: arguments.steps4_dir,
        8: arguments.steps8_dir,
        16: arguments.steps16_dir,
    }
    output_directory = arguments.output_dir.resolve()
    plots_directory = output_directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)

    summary, curves = load_conditions(directories)
    aggregate = aggregate_conditions(summary)
    differences, comparisons = paired_reward_analysis(
        summary, bootstrap_samples=arguments.bootstrap_samples
    )
    probe_frame = pd.read_csv(arguments.probe_observations)
    probe = probe_frame.to_numpy(dtype=np.float32)
    activity, operations = analyse_spikes(
        directories, probe=probe, batch_size=arguments.batch_size
    )

    summary.to_csv(output_directory / "summary_by_seed.csv", index=False)
    curves.to_csv(output_directory / "learning_curves.csv", index=False)
    episode_frames = []
    configuration_directory = output_directory / "configurations"
    configuration_directory.mkdir(parents=True, exist_ok=True)
    for steps, directory in directories.items():
        episodes = pd.read_csv(directory / "evaluation_episodes.csv")
        episodes.insert(0, "steps", steps)
        counts = episodes.groupby("seed").size()
        if len(counts) != 20 or not (counts == 50).all():
            raise ValueError(
                f"Expected 50 final episodes for each of 20 seeds at {steps} steps"
            )
        episode_frames.append(episodes)
        with (directory / "experiment_config.json").open(encoding="utf-8") as file:
            configuration_payload = json.load(file)
        with (configuration_directory / f"steps_{steps}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(configuration_payload, file, indent=2)
            file.write("\n")
    pd.concat(episode_frames, ignore_index=True).to_csv(
        output_directory / "evaluation_episodes.csv", index=False
    )
    aggregate.to_csv(output_directory / "aggregate_summary.csv", index=False)
    differences.to_csv(output_directory / "paired_reward_differences.csv", index=False)
    comparisons.to_csv(output_directory / "paired_reward_comparisons.csv", index=False)
    activity.to_csv(output_directory / "spike_activity_by_model.csv", index=False)
    operations.to_csv(output_directory / "operation_estimates_by_model.csv", index=False)

    configure_plot_style()
    plot_learning_curves(curves, plots_directory / "learning_curves_by_steps.png")
    plot_computational_cost(summary, plots_directory / "cpu_cost_by_steps.png")

    print(f"Timestep analysis complete: {output_directory}", flush=True)


if __name__ == "__main__":
    main()
