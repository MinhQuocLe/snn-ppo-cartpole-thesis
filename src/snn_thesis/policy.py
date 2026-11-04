"""Stable-Baselines3 PPO policy components using LIF spiking layers."""

from __future__ import annotations

from collections.abc import Sequence

import snntorch as snn
import torch
from snntorch import surrogate
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


DEFAULT_CARTPOLE_SCALES = (2.4, 3.0, 0.2095, 3.5)
SNN_INPUT_ENCODINGS = ("gaussian", "direct")


class DirectCurrentEncoder(nn.Module):
    """Pass continuous observations directly to the first learned SNN layer."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.output_dim = self.observation_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                "observations must have shape "
                f"[batch, {self.observation_dim}], got {tuple(observations.shape)}"
            )
        return observations


class GaussianPopulationEncoder(nn.Module):
    """Encode each scalar observation with Gaussian receptive fields.

    CartPole position and angle use their termination thresholds as scales.
    The velocity scales are explicit experimental assumptions because those
    observation dimensions have unbounded declared ranges in Gymnasium.
    """

    def __init__(
        self,
        observation_dim: int,
        population_size: int = 4,
        scales: Sequence[float] = DEFAULT_CARTPOLE_SCALES,
        sigma: float = 0.5,
    ) -> None:
        super().__init__()
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if len(scales) != observation_dim:
            raise ValueError(
                f"Expected {observation_dim} observation scales, got {len(scales)}"
            )
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.observation_dim = observation_dim
        self.population_size = population_size
        self.output_dim = observation_dim * population_size
        self.sigma = float(sigma)

        self.register_buffer(
            "scales",
            torch.as_tensor(scales, dtype=torch.float32).reshape(1, observation_dim),
        )
        self.register_buffer(
            "centres",
            torch.linspace(-1.0, 1.0, population_size, dtype=torch.float32).reshape(
                1, 1, population_size
            ),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                "observations must have shape "
                f"[batch, {self.observation_dim}], got {tuple(observations.shape)}"
            )

        normalised = torch.clamp(observations / self.scales, -1.0, 1.0)
        distance = normalised.unsqueeze(-1) - self.centres
        activations = torch.exp(-0.5 * (distance / self.sigma).square())
        return activations.flatten(start_dim=1)


class SpikingBranch(nn.Module):
    """A feed-forward LIF network unrolled over internal simulation time."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        num_steps: int,
        beta: float,
        threshold: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.num_steps = int(num_steps)

        spike_gradient = surrogate.fast_sigmoid(slope=surrogate_slope)
        layer_sizes = (input_dim, *self.hidden_sizes)
        self.linear_layers = nn.ModuleList(
            nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            for index in range(len(self.hidden_sizes))
        )
        self.lif_layers = nn.ModuleList(
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_gradient,
                reset_mechanism="subtract",
                # Keep snnTorch's training-time behaviour explicit: a spike's
                # subtractive reset is applied on the following internal tick.
                reset_delay=True,
            )
            for _ in self.hidden_sizes
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_sizes[-1]

    def forward(self, encoded_observations: torch.Tensor) -> torch.Tensor:
        batch_size = encoded_observations.shape[0]
        membrane_states = [
            encoded_observations.new_zeros((batch_size, hidden_size))
            for hidden_size in self.hidden_sizes
        ]
        output_spike_sum = encoded_observations.new_zeros(
            (batch_size, self.output_dim)
        )

        for _ in range(self.num_steps):
            spikes = encoded_observations
            for layer_index, (linear, lif) in enumerate(
                zip(self.linear_layers, self.lif_layers, strict=True)
            ):
                current = linear(spikes)
                spikes, membrane_states[layer_index] = lif(
                    current, membrane_states[layer_index]
                )
            output_spike_sum = output_spike_sum + spikes

        return output_spike_sum / float(self.num_steps)


class SNNMlpExtractor(nn.Module):
    """Stable-Baselines3-compatible actor/critic latent feature extractor."""

    def __init__(
        self,
        feature_dim: int,
        hidden_sizes: Sequence[int] = (59, 59),
        input_encoding: str = "gaussian",
        population_size: int = 4,
        population_sigma: float = 0.5,
        observation_scales: Sequence[float] = DEFAULT_CARTPOLE_SCALES,
        num_steps: int = 8,
        beta: float = 0.9,
        threshold: float = 0.5,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()
        if input_encoding == "gaussian":
            self.encoder = GaussianPopulationEncoder(
                observation_dim=feature_dim,
                population_size=population_size,
                scales=observation_scales,
                sigma=population_sigma,
            )
        elif input_encoding == "direct":
            self.encoder = DirectCurrentEncoder(observation_dim=feature_dim)
        else:
            raise ValueError(
                f"input_encoding must be one of {SNN_INPUT_ENCODINGS}, "
                f"got {input_encoding!r}"
            )
        branch_kwargs = {
            "input_dim": self.encoder.output_dim,
            "hidden_sizes": hidden_sizes,
            "num_steps": num_steps,
            "beta": beta,
            "threshold": threshold,
            "surrogate_slope": surrogate_slope,
        }
        self.policy_net = SpikingBranch(**branch_kwargs)
        self.value_net = SpikingBranch(**branch_kwargs)
        self.latent_dim_pi = self.policy_net.output_dim
        self.latent_dim_vf = self.value_net.output_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return self.policy_net(encoded), self.value_net(encoded)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.encoder(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.encoder(features))


class SNNActorCriticPolicy(ActorCriticPolicy):
    """PPO actor-critic policy with separate surrogate-gradient SNN branches."""

    def __init__(
        self,
        *args,
        snn_hidden_sizes: Sequence[int] = (59, 59),
        input_encoding: str = "gaussian",
        population_size: int = 4,
        population_sigma: float = 0.5,
        observation_scales: Sequence[float] = DEFAULT_CARTPOLE_SCALES,
        num_steps: int = 8,
        beta: float = 0.9,
        threshold: float = 0.5,
        surrogate_slope: float = 25.0,
        **kwargs,
    ) -> None:
        self.snn_hidden_sizes = tuple(int(size) for size in snn_hidden_sizes)
        if input_encoding not in SNN_INPUT_ENCODINGS:
            raise ValueError(
                f"input_encoding must be one of {SNN_INPUT_ENCODINGS}, "
                f"got {input_encoding!r}"
            )
        self.input_encoding = input_encoding
        self.population_size = int(population_size)
        self.population_sigma = float(population_sigma)
        self.observation_scales = tuple(float(scale) for scale in observation_scales)
        self.num_steps = int(num_steps)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)

        # The custom extractor replaces SB3's ordinary MLP.  SB3 still creates
        # and initialises the final categorical-action and scalar-value heads.
        super().__init__(
            *args,
            net_arch=[],
            activation_fn=nn.Identity,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SNNMlpExtractor(
            feature_dim=self.features_dim,
            hidden_sizes=self.snn_hidden_sizes,
            input_encoding=self.input_encoding,
            population_size=self.population_size,
            population_sigma=self.population_sigma,
            observation_scales=self.observation_scales,
            num_steps=self.num_steps,
            beta=self.beta,
            threshold=self.threshold,
            surrogate_slope=self.surrogate_slope,
        ).to(self.device)
