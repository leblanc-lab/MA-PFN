"""Model definitions for the minimal MA-PFN versus PFN comparison."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MAPELoss(nn.Module):
    """Mean absolute percentage error used for both models."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((target - prediction) / (target + self.eps)))


class ParticleLevelLinear(nn.Module):
    """A dense layer applied independently to every particle."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        scale = input_dim**-0.5
        self.weights = nn.Parameter(
            torch.rand(input_dim, output_dim) * 2 * scale - scale
        )
        self.bias = nn.Parameter(torch.rand(output_dim) * 2 * scale - scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(inputs, self.weights) + self.bias


class MAPFN(nn.Module):
    """Metric-aware PFN with three exact structural metric properties."""

    def __init__(
        self, input_dim: int, latent_dim: int, phi_hidden_dim: int, f_hidden_dim: int
    ) -> None:
        super().__init__()
        self.phi_fc1 = ParticleLevelLinear(input_dim, phi_hidden_dim)
        self.phi_fc2 = ParticleLevelLinear(phi_hidden_dim, phi_hidden_dim)
        self.phi_fc3 = ParticleLevelLinear(phi_hidden_dim, latent_dim)

        # No biases: zero latent difference must map to zero.
        self.f_diff1 = nn.Linear(latent_dim, f_hidden_dim, bias=False)
        self.f_diff2 = nn.Linear(f_hidden_dim, f_hidden_dim, bias=False)
        self.f_diff3 = nn.Linear(f_hidden_dim, f_hidden_dim, bias=False)
        self.f_diff4 = nn.Linear(f_hidden_dim, 1, bias=False)

        self.f_sum1 = nn.Linear(latent_dim, f_hidden_dim)
        self.f_sum2 = nn.Linear(f_hidden_dim, f_hidden_dim)
        self.f_sum3 = nn.Linear(f_hidden_dim, f_hidden_dim)
        self.f_sum4 = nn.Linear(f_hidden_dim, 1)

    def _particle_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode particles without learning the event-identity tag."""

        particle_mask = (inputs[..., :3].abs().sum(dim=-1) > 0).unsqueeze(-1)
        if inputs.shape[-1] == 3:
            kinematics = F.pad(inputs, (0, 1))
        elif inputs.shape[-1] == 4:
            kinematics = inputs.clone()
            kinematics[..., 3] = 0.0
        else:
            raise ValueError("Expected three kinematic or four tagged features")
        features = F.relu(self.phi_fc1(kinematics))
        features = F.relu(self.phi_fc2(features))
        return self.phi_fc3(features) * particle_mask

    def encode_events(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode each unpaired event once for reuse across many pairs."""

        return torch.sum(self._particle_features(inputs), dim=1)

    def event_latents(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape[-1] != 4:
            raise ValueError("Paired inputs must include an event-identity tag")
        event_id = inputs[..., 3:4]
        features = self._particle_features(inputs)
        first = torch.sum(features * (event_id == -1), dim=1)
        second = torch.sum(features * (event_id == 1), dim=1)
        return first, second

    def pairwise_from_latents(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        """Score aligned pairs of cached event latents."""

        difference = torch.abs(first - second)
        total = first + second

        difference = F.relu(self.f_diff1(difference))
        difference = F.relu(self.f_diff2(difference))
        difference = F.relu(self.f_diff3(difference))
        difference = self.f_diff4(difference)

        total = F.relu(self.f_sum1(total))
        total = F.relu(self.f_sum2(total))
        total = F.relu(self.f_sum3(total))
        total = self.f_sum4(total)
        prediction = torch.abs(difference * total)[:, 0]

        # This also removes tiny floating-point remnants for identical latents.
        identical = torch.isclose(first, second, rtol=1e-5, atol=1e-6).all(dim=1)
        return torch.where(identical, torch.zeros_like(prediction), prediction)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pairwise_from_latents(*self.event_latents(inputs))


class PFN(nn.Module):
    """The original unconstrained PFN regression architecture."""

    def __init__(
        self, input_dim: int, latent_dim: int, phi_hidden_dim: int, f_hidden_dim: int
    ) -> None:
        super().__init__()
        self.phi_fc1 = ParticleLevelLinear(input_dim, phi_hidden_dim)
        self.phi_fc2 = ParticleLevelLinear(phi_hidden_dim, phi_hidden_dim)
        self.phi_fc3 = ParticleLevelLinear(phi_hidden_dim, latent_dim)
        self.f_fc1 = nn.Linear(latent_dim, f_hidden_dim)
        self.f_fc2 = nn.Linear(f_hidden_dim, f_hidden_dim)
        self.f_fc3 = nn.Linear(f_hidden_dim, f_hidden_dim)
        self.f_fc4 = nn.Linear(f_hidden_dim, 1)

    def _particle_features(self, inputs: torch.Tensor) -> torch.Tensor:
        particle_mask = (inputs[..., :3].abs().sum(dim=-1) > 0).unsqueeze(-1)
        features = F.relu(self.phi_fc1(inputs))
        features = F.relu(self.phi_fc2(features))
        return self.phi_fc3(features) * particle_mask

    def encode_events(self, inputs: torch.Tensor, event_id: float) -> torch.Tensor:
        """Encode unpaired events for one of the two tagged pair roles."""

        if inputs.shape[-1] != 3:
            raise ValueError("Unpaired event inputs must have three kinematic features")
        event_tag = torch.full(
            (*inputs.shape[:-1], 1),
            event_id,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        tagged = torch.cat((inputs, event_tag), dim=-1)
        return torch.sum(self._particle_features(tagged), dim=1)

    def pairwise_from_latents(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        """Score cached PFN latents from the first and second tagged roles."""

        return self._head(first + second)

    def _head(self, latent: torch.Tensor) -> torch.Tensor:
        latent = F.relu(self.f_fc1(latent))
        latent = F.relu(self.f_fc2(latent))
        latent = F.relu(self.f_fc3(latent))
        return self.f_fc4(latent)[:, 0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = torch.sum(self._particle_features(inputs), dim=1)
        return self._head(latent)


__all__ = ["MAPELoss", "MAPFN", "PFN", "ParticleLevelLinear"]
