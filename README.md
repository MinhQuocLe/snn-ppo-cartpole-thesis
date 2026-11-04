# PPO comparison of ANN and spiking policies on CartPole

This repository contains the code and results for a controlled comparison of
an artificial neural network (ANN) and leaky integrate-and-fire spiking neural networks
(SNNs). Every policy was trained with PPO on CartPole-v1.

The study tested five conditions with 20 independent training seeds each, giving 100
trained policies in total.

## Main results

| Condition | Final reward, mean +/- SD | Successful runs |
|---|---:|---:|
| ANN, 64--64 | 499.84 +/- 0.71 | 20/20 |
| Gaussian-input SNN, 8 steps, 59--59 | 446.87 +/- 89.56 | 12/20 |
| Direct-input SNN, 4 steps, 64--64 | 480.40 +/- 39.76 | 16/20 |
| Direct-input SNN, 8 steps, 64--64 | 495.90 +/- 9.87 | 19/20 |
| Direct-input SNN, 16 steps, 64--64 | 469.69 +/- 59.79 | 15/20 |

A run was counted as successful when its mean reward was at least 475 across 50 fixed-seed
final evaluation episodes. The ANN was the most reliable condition. Among the tested
SNNs, direct input with eight internal steps gave the strongest final result.

An internal step is one update of the LIF layers inside a single policy decision. For
example, eight internal steps mean that the same CartPole observation is processed eight
times before the policy chooses one action. CartPole still advances only once.

## Repository structure

```text
src/snn_thesis/   model, training, evaluation, and analysis code
results/          final CSV/JSON data and key plots
```

Useful starting points:

- [`src/snn_thesis/policy.py`](src/snn_thesis/policy.py): input encoders, LIF layers,
  membrane reset, internal steps, and PPO-compatible policy.
- [`src/snn_thesis/experiment.py`](src/snn_thesis/experiment.py): training and final
  evaluation pipeline.
- [`results/summary.csv`](results/summary.csv): per-seed ANN and Gaussian-SNN results.
- [`results/timestep_analysis/summary_by_seed.csv`](results/timestep_analysis/summary_by_seed.csv):
  per-seed direct-input SNN results at 4, 8, and 16 internal steps.
- [`results/spike_analysis/spike_activity_by_model.csv`](results/spike_analysis/spike_activity_by_model.csv):
  final-model firing activity.

## Install

The project uses Python 3.13 and Poetry.

```powershell
poetry install
poetry run snn-run --help
```

## Run the experiments

Run the ANN and Gaussian-input SNN comparison:

```powershell
poetry run snn-run `
  --models ann snn `
  --snn-input-encoding gaussian `
  --snn-hidden-sizes 59,59 `
  --snn-steps 8 `
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 `
  --total-timesteps 50000 `
  --device cpu `
  --output-dir artifacts/cpu-20-seeds
```

Run the direct-input eight-step SNN:

```powershell
poetry run snn-run `
  --models snn `
  --snn-input-encoding direct `
  --snn-hidden-sizes 64,64 `
  --snn-steps 8 `
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 `
  --total-timesteps 50000 `
  --device cpu `
  --output-dir artifacts/cpu-20-seeds-direct
```

For the 4- and 16-step conditions, repeat the direct-input command above and change
`--snn-steps` to `4` or `16`. Use the matching output directories
`artifacts/cpu-20-seeds-direct-steps4` and
`artifacts/cpu-20-seeds-direct-steps16`. The command above is the 8-step condition.

Use `--resume` after an interrupted run. Complete working runs are written to the ignored
`artifacts/` directory. The final data and key plots are kept under `results/`.

## Limits of the comparison

- The results cover CartPole-v1 and the tested PPO/SNN designs only.
- The Gaussian and direct SNNs also used different hidden widths, so this is a comparison
  of complete designs rather than an encoding-only test.
- Equal parameter counts do not mean equal computation. SNN weights are reused across
  internal steps.
- CPU time and estimated operations do not measure energy or speed on neuromorphic
  hardware.
