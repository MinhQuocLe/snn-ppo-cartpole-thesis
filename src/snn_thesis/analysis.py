"""Create statistical summaries and plots for an experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ("ANN", "SNN")
MODEL_COLOURS = {"ANN": "#0072B2", "SNN": "#D55E00"}
SOLVED_REWARD = 475.0
MAXIMUM_REWARD = 500.0


# ---------------------------------------------------------------------------
# Descriptive and paired statistics
# ---------------------------------------------------------------------------


def aggregate_runs(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_mean_reward",
        "final_median_reward",
        "solved_episode_fraction",
        "first_solved_timestep",
        "sustained_solved_timestep",
        "training_time_excluding_eval_seconds",
        "inference_latency_median_ms",
        "inference_latency_p95_ms",
        "trainable_parameters",
    ]
    available = [metric for metric in metrics if metric in summary.columns]
    aggregate = (
        summary.groupby("model", sort=False)[available]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        column
        if isinstance(column, str)
        else "_".join(part for part in column if part)
        for column in aggregate.columns
    ]
    return aggregate


def paired_differences(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_mean_reward",
        "first_solved_timestep",
        "training_time_excluding_eval_seconds",
        "inference_latency_median_ms",
    ]
    rows: list[dict[str, float | int]] = []
    for metric in metrics:
        if metric not in summary.columns:
            continue
        pivot = summary.pivot(index="seed", columns="model", values=metric)
        if not {"ANN", "SNN"}.issubset(pivot.columns):
            continue
        for seed, values in pivot[["ANN", "SNN"]].dropna().iterrows():
            rows.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "ann": float(values["ANN"]),
                    "snn": float(values["SNN"]),
                    "snn_minus_ann": float(values["SNN"] - values["ANN"]),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_paired_differences(
    paired: pd.DataFrame,
    bootstrap_samples: int = 20_000,
    seed: int = 2026,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "paired_seeds",
                "mean_snn_minus_ann",
                "median_snn_minus_ann",
                "bootstrap_95_ci_low",
                "bootstrap_95_ci_high",
            ]
        )

    random_generator = np.random.default_rng(seed)
    rows = []
    for metric, metric_rows in paired.groupby("metric"):
        differences = metric_rows["snn_minus_ann"].to_numpy(dtype=np.float64)
        sampled_indices = random_generator.integers(
            0, len(differences), size=(bootstrap_samples, len(differences))
        )
        bootstrapped_means = differences[sampled_indices].mean(axis=1)
        rows.append(
            {
                "metric": metric,
                "paired_seeds": len(differences),
                "mean_snn_minus_ann": float(differences.mean()),
                "median_snn_minus_ann": float(np.median(differences)),
                "bootstrap_95_ci_low": float(np.quantile(bootstrapped_means, 0.025)),
                "bootstrap_95_ci_high": float(np.quantile(bootstrapped_means, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_learning_curves(curves: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for model in MODEL_ORDER:
        model_rows = curves[curves["model"] == model]
        if model_rows.empty:
            continue
        aggregate = (
            model_rows.groupby("timesteps")["mean_eval_reward"]
            .agg(["mean", "std"])
            .reset_index()
        )
        aggregate["std"] = aggregate["std"].fillna(0.0)
        x = aggregate["timesteps"].to_numpy(dtype=float)
        mean = aggregate["mean"].to_numpy(dtype=float)
        standard_deviation = aggregate["std"].to_numpy(dtype=float)
        colour = MODEL_COLOURS[model]
        axis.plot(x, mean, linewidth=2, color=colour, label=model)
        axis.fill_between(
            x,
            np.clip(mean - standard_deviation, 0, MAXIMUM_REWARD),
            np.clip(mean + standard_deviation, 0, MAXIMUM_REWARD),
            color=colour,
            alpha=0.18,
        )

    axis.axhline(
        SOLVED_REWARD,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="Solved threshold (475)",
    )
    axis.set(
        title="PPO learning performance on CartPole-v1",
        xlabel="Environment interactions",
        ylabel="Mean evaluation reward",
        ylim=(0, 510),
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.savefig(output_path)
    plt.close(figure)


def generate_analysis_artifacts(
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory = Path(output_directory)
    plots_directory = output_directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    aggregate = aggregate_runs(summary)
    aggregate.to_csv(output_directory / "aggregate_summary.csv", index=False)

    paired = paired_differences(summary)
    paired.to_csv(output_directory / "paired_differences.csv", index=False)
    bootstrap = bootstrap_paired_differences(paired)
    bootstrap.to_csv(output_directory / "paired_bootstrap_summary.csv", index=False)

    plot_learning_curves(curves, plots_directory / "learning_curves_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    generate_analysis_artifacts(
        summary=pd.read_csv(output_directory / "summary.csv"),
        curves=pd.read_csv(output_directory / "eval_curve.csv"),
        output_directory=output_directory,
    )


if __name__ == "__main__":
    main()
