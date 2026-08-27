"""Model definitions for the minimal MA-PFN versus PFN comparison."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HybridEMDLoss(nn.Module):
    """MAPE, MAE, or MAPE plus a dimensionless normalized-MAE term.

    This is the loss used by the full training workflow. Keeping its
    components available lets the tutorial optimize the hybrid objective while
    reporting the two physically interpretable error measures separately.
    """

    def __init__(
        self,
        mae_weight: float = 0.0,
        mae_scale: float = 90.0,
        eps: float = 1e-8,
        loss: str = "hybrid",
    ) -> None:
        super().__init__()
        if loss not in {"mape", "mae", "hybrid"}:
            raise ValueError("loss must be mape, mae, or hybrid")
        if mae_weight < 0:
            raise ValueError("mae_weight must be non-negative")
        if mae_scale <= 0:
            raise ValueError("mae_scale must be positive")
        self.loss = loss
        self.mae_weight = mae_weight
        self.mae_scale = mae_scale
        self.eps = eps

    @property
    def description(self) -> str:
        if self.loss == "mape":
            return "MAPE"
        if self.loss == "mae":
            return "MAE"
        return f"MAPE + {self.mae_weight:g} * MAE / {self.mae_scale:g} GeV"

    def objective_from_metrics(
        self, mape: torch.Tensor | float, mae: torch.Tensor | float
    ) -> torch.Tensor | float:
        if self.loss == "mape":
            return mape
        if self.loss == "mae":
            return mae
        return mape + self.mae_weight * mae / self.mae_scale

    def components(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        absolute_error = torch.abs(target - prediction)
        mape = torch.mean(absolute_error / (target.abs() + self.eps))
        mae = torch.mean(absolute_error)
        objective = self.objective_from_metrics(mape, mae)
        return objective, mape, mae

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(prediction, target)[0]


class MAPELoss(HybridEMDLoss):
    """Backward-compatible pure-MAPE specialization."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__(eps=eps, loss="mape")


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
    """Exchange-symmetrized joint-head MA-PFN used by the release model.

    The shared particle encoder produces one latent vector per event.  A joint
    head sees the latent sum and signed difference in both event orientations.
    Averaging those two evaluations makes exchange symmetry exact without
    discarding sign information.  The mean absolute latent separation gives
    exact zero self-distance, and softplus makes predictions non-negative.
    """

    def __init__(
        self, input_dim: int, latent_dim: int, phi_hidden_dim: int, f_hidden_dim: int
    ) -> None:
        super().__init__()
        self.phi_fc1 = ParticleLevelLinear(input_dim, phi_hidden_dim)
        self.phi_fc2 = ParticleLevelLinear(phi_hidden_dim, phi_hidden_dim)
        self.phi_fc3 = ParticleLevelLinear(phi_hidden_dim, latent_dim)

        self.f_joint1 = nn.Linear(2 * latent_dim, f_hidden_dim)
        self.f_joint2 = nn.Linear(f_hidden_dim, f_hidden_dim)
        self.f_joint3 = nn.Linear(f_hidden_dim, f_hidden_dim)
        self.f_joint4 = nn.Linear(f_hidden_dim, 1)

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

        latent_sum = first + second
        latent_difference = first - second
        identical = torch.isclose(
            first, second, rtol=1e-5, atol=1e-6
        ).all(dim=1)
        symmetric_log_scale = 0.5 * (
            self.joint_head(latent_sum, latent_difference)
            + self.joint_head(latent_sum, -latent_difference)
        )
        latent_separation = torch.mean(torch.abs(latent_difference), dim=1)
        prediction = latent_separation * F.softplus(symmetric_log_scale)
        return torch.where(identical, torch.zeros_like(prediction), prediction)

    def joint_head(
        self, latent_sum: torch.Tensor, latent_difference: torch.Tensor
    ) -> torch.Tensor:
        """Return the learned log scale for one signed event orientation."""

        features = torch.cat((latent_sum, latent_difference), dim=1)
        features = F.relu(self.f_joint1(features))
        features = F.relu(self.f_joint2(features))
        features = F.relu(self.f_joint3(features))
        return self.f_joint4(features)[:, 0]

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


__all__ = [
    "HybridEMDLoss",
    "MAPELoss",
    "MAPFN",
    "PFN",
    "ParticleLevelLinear",
]
