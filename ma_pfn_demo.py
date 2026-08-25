# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Minimal MA-PFN versus PFN example
#
# This is the single source for both the command-line example and the paired
# notebook. It trains the metric-aware particle flow network (MA-PFN) and a
# stock PFN on identical event-pair splits, evaluates both on held-out data,
# and benchmarks non-negativity, identity, symmetry, and triangle inequality.

# %%
"""Train and benchmark a minimal MA-PFN and stock-PFN comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


MODEL_LABELS = {"ma_pfn": "MA-PFN", "pfn": "PFN"}
MODEL_COLORS = {"ma_pfn": "#0072B2", "pfn": "#D55E00"}


@dataclass
class WorkflowConfig:
    """All settings needed to reproduce a training and benchmark run."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("results")
    stage: str = "all"
    epochs: int = 500
    patience: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-4
    num_workers: int = 0
    seed: int = 12345
    latent_dim: int = 64
    phi_hidden_dim: int = 100
    f_hidden_dim: int = 100
    max_train_pairs: int | None = None
    max_val_pairs: int | None = None
    max_test_pairs: int | None = 100_000
    metric_samples: int = 20_000
    tolerance: float = 1e-3
    device: str = "auto"
    preload: bool = False
    make_plots: bool = True


# %% [markdown]
# ## Models
#
# The stock PFN embeds every tagged particle and sums all particle embeddings.
# MA-PFN instead embeds particle kinematics with a shared map, pools the two
# events separately, and uses exchange-invariant sum and absolute-difference
# representations. A bias-free difference branch and an absolute-valued output
# enforce zero self-distance and non-negativity. Triangle inequality is not
# imposed and is therefore tested empirically.

# %%
class MAPELoss(nn.Module):
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

    def event_latents(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        particle_mask = (inputs[..., :3].abs().sum(dim=-1) > 0).unsqueeze(-1)
        event_id = inputs[..., 3:4]

        # The tag chooses the event pool but is not a learned kinematic feature.
        kinematics = inputs.clone()
        kinematics[..., 3] = 0.0
        features = F.relu(self.phi_fc1(kinematics))
        features = F.relu(self.phi_fc2(features))
        features = self.phi_fc3(features) * particle_mask

        first = torch.sum(features * (event_id == -1), dim=1)
        second = torch.sum(features * (event_id == 1), dim=1)
        return first, second

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = self.event_latents(inputs)
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
        identical = torch.isclose(
            first, second, rtol=1e-5, atol=1e-6
        ).all(dim=1)
        return torch.where(identical, torch.zeros_like(prediction), prediction)


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        particle_mask = (inputs[..., :3].abs().sum(dim=-1) > 0).unsqueeze(-1)
        features = F.relu(self.phi_fc1(inputs))
        features = F.relu(self.phi_fc2(features))
        features = self.phi_fc3(features) * particle_mask
        latent = features.sum(dim=1)
        latent = F.relu(self.f_fc1(latent))
        latent = F.relu(self.f_fc2(latent))
        latent = F.relu(self.f_fc3(latent))
        return self.f_fc4(latent)[:, 0]


def make_model(name: str, config: WorkflowConfig) -> nn.Module:
    model_class: type[nn.Module] = MAPFN if name == "ma_pfn" else PFN
    return model_class(
        input_dim=4,
        latent_dim=config.latent_dim,
        phi_hidden_dim=config.phi_hidden_dim,
        f_hidden_dim=config.f_hidden_dim,
    )


# %% [markdown]
# ## Data and training
#
# Each split contains `SPLIT_features.npy` and `SPLIT_targets.npy`. Features
# have shape `(pairs, particles, 4)` and store `(pT, eta, phi, event_id)`;
# targets have shape `(pairs,)` and store exact EMD in GeV. Memory mapping keeps
# the full release usable without loading every pair into RAM.

# %%
class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        limit: int | None,
        seed: int,
        preload: bool = False,
    ) -> None:
        self.features_path = data_dir / f"{split}_features.npy"
        self.targets_path = data_dir / f"{split}_targets.npy"
        if not self.features_path.is_file() or not self.targets_path.is_file():
            raise FileNotFoundError(
                f"Expected {self.features_path.name} and {self.targets_path.name}"
            )
        self.features = np.load(self.features_path, mmap_mode="r")
        self.targets = np.load(self.targets_path, mmap_mode="r")
        if self.features.ndim != 3 or self.features.shape[-1] != 4:
            raise ValueError(
                f"{self.features_path} must have shape (pairs, particles, 4)"
            )
        if self.targets.ndim != 1 or len(self.features) != len(self.targets):
            raise ValueError(f"Feature/target shapes do not match for {split}")

        self.indices: np.ndarray | None = None
        if limit is not None and limit > 0 and limit < len(self.targets):
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(len(self.targets), limit, replace=False))

        self.preloaded_features: torch.Tensor | None = None
        self.preloaded_targets: torch.Tensor | None = None
        if preload:
            selected = slice(None) if self.indices is None else self.indices
            print(f"Preloading {split} split as float32", flush=True)
            features = np.array(self.features[selected], dtype=np.float32, copy=True)
            targets = np.array(self.targets[selected], dtype=np.float32, copy=True)
            self.preloaded_features = torch.from_numpy(features)
            self.preloaded_targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.targets) if self.indices is None else len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.preloaded_features is not None and self.preloaded_targets is not None:
            return self.preloaded_features[index], self.preloaded_targets[index]
        source_index = index if self.indices is None else int(self.indices[index])
        # A writable float32 copy avoids NumPy memmap warnings in torch.from_numpy.
        features = np.array(self.features[source_index], dtype=np.float32, copy=True)
        target = np.float32(self.targets[source_index])
        return torch.from_numpy(features), torch.tensor(target)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    examples = 0
    with torch.inference_mode():
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = loss_fn(model(features), target)
            total += loss.item() * len(features)
            examples += len(features)
    return total / examples


def train_one_model(
    name: str,
    config: WorkflowConfig,
    train_data: PairDataset,
    val_data: PairDataset,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    set_seed(config.seed)
    model = make_model(name, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = MAPELoss()

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers:
        loader_options["persistent_workers"] = True
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data, shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    run_dir = config.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training {MODEL_LABELS[name]} ({parameter_count:,} parameters) on {device}",
        flush=True,
    )

    history = {"train_mape": [], "val_mape": []}
    best_loss = float("inf")
    best_epoch = -1
    started = time.time()
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        examples = 0
        for features, target in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(features)
            examples += len(features)

        train_loss = total / examples
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history["train_mape"].append(train_loss)
        history["val_mape"].append(val_loss)
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "format_version": 1,
                "architecture": name,
                "model_kwargs": {
                    "input_dim": 4,
                    "latent_dim": config.latent_dim,
                    "phi_hidden_dim": config.phi_hidden_dim,
                    "f_hidden_dim": config.f_hidden_dim,
                },
                "model_state_dict": model.state_dict(),
            }
            torch.save(checkpoint, run_dir / "best_model.pt")

        elapsed = (time.time() - started) / 60
        marker = " *" if improved else ""
        print(
            f"  epoch {epoch + 1:03d}/{config.epochs}: "
            f"train={train_loss:.6f}, val={val_loss:.6f}, {elapsed:.1f} min{marker}",
            flush=True,
        )
        if config.patience and epoch - best_epoch >= config.patience:
            print(f"  early stopping after epoch {epoch + 1}", flush=True)
            break

    history.update(
        {
            "model": name,
            "parameter_count": parameter_count,
            "epochs_completed": len(history["train_mape"]),
            "best_epoch": best_epoch + 1,
            "best_val_mape": best_loss,
        }
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return load_checkpoint(run_dir / "best_model.pt", device), history


def load_checkpoint(path: Path, device: torch.device) -> nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was added.
        checkpoint = torch.load(path, map_location="cpu")
    name = checkpoint["architecture"]
    model_class: type[nn.Module] = MAPFN if name == "ma_pfn" else PFN
    model = model_class(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


# %% [markdown]
# ## Held-out accuracy and metric-property benchmarks
#
# Test events are reconstructed from the complete combinations-ordered test
# split. This keeps the example self-contained: no private generator-level
# event file is needed for self-distance, reversed-pair, or triplet tests.

# %%
def subset_indices(length: int, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit <= 0 or limit >= length:
        return np.arange(length)
    return np.sort(np.random.default_rng(seed).choice(length, limit, replace=False))


def predict_features(
    model: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    prediction = np.empty(len(indices), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            batch = np.array(features[indices[start:stop]], dtype=np.float32, copy=True)
            tensor = torch.from_numpy(batch).to(device, non_blocking=True)
            prediction[start:stop] = model(tensor).cpu().numpy()
    return prediction


def reconstruct_split_events(data_dir: Path, split: str = "test") -> np.ndarray:
    path = data_dir / f"{split}_features.npy"
    features = np.load(path, mmap_mode="r")
    pair_count = len(features)
    event_count = (1 + math.isqrt(1 + 8 * pair_count)) // 2
    if event_count * (event_count - 1) // 2 != pair_count:
        raise ValueError(
            f"{path} is not a complete combinations-ordered pair split"
        )
    if features.shape[1] % 2:
        raise ValueError("Paired events must have equal padded particle counts")
    particles = features.shape[1] // 2
    first_rows = np.asarray(features[: min(event_count - 1, 8), :, 3])
    if not (
        np.all(first_rows[:, :particles] == -1)
        and np.all(first_rows[:, particles:] == 1)
    ):
        raise ValueError("Expected the first event tagged -1 and the second +1")

    events = np.empty((event_count, particles, 3), dtype=np.float32)
    events[0] = features[0, :particles, :3]
    events[1:] = features[: event_count - 1, particles:, :3]
    return events


def pair_events(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    first_tag = -np.ones((*first.shape[:-1], 1), dtype=np.float32)
    second_tag = np.ones((*second.shape[:-1], 1), dtype=np.float32)
    return np.concatenate(
        (
            np.concatenate((first, first_tag), axis=-1),
            np.concatenate((second, second_tag), axis=-1),
        ),
        axis=1,
    )


def predict_event_pairs(
    model: nn.Module,
    events: np.ndarray,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    prediction = np.empty(len(first_indices), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(first_indices), batch_size):
            stop = min(start + batch_size, len(first_indices))
            batch = pair_events(
                events[first_indices[start:stop]], events[second_indices[start:stop]]
            )
            tensor = torch.from_numpy(batch).to(device, non_blocking=True)
            prediction[start:stop] = model(tensor).cpu().numpy()
    return prediction


def distinct_rows(
    event_count: int, sample_count: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    if event_count < width:
        raise ValueError(f"Need at least {width} events, found {event_count}")
    rows = rng.integers(0, event_count, size=(sample_count, width))
    duplicate = np.ones(sample_count, dtype=bool)
    while np.any(duplicate):
        duplicate = np.any(np.diff(np.sort(rows, axis=1), axis=1) == 0, axis=1)
        rows[duplicate] = rng.integers(
            0, event_count, size=(int(duplicate.sum()), width)
        )
    return rows


def regression_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    target_variance = np.sum((target - np.mean(target)) ** 2)
    return {
        "mape": float(np.mean(np.abs(residual / target))),
        "mae_gev": float(np.mean(np.abs(residual))),
        "rmse_gev": float(np.sqrt(np.mean(residual**2))),
        "r_squared": float(1 - np.sum(residual**2) / target_variance),
    }


def benchmark_models(
    models: dict[str, nn.Module], config: WorkflowConfig, device: torch.device
) -> tuple[dict, dict[str, np.ndarray]]:
    test_features = np.load(config.data_dir / "test_features.npy", mmap_mode="r")
    test_targets = np.load(config.data_dir / "test_targets.npy", mmap_mode="r")
    test_indices = subset_indices(
        len(test_targets), config.max_test_pairs, config.seed + 20
    )
    target = np.asarray(test_targets[test_indices], dtype=np.float32)

    arrays: dict[str, np.ndarray] = {
        "test_indices": test_indices,
        "target": target,
    }
    accuracy: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        print(f"Predicting {len(test_indices):,} held-out pairs with {MODEL_LABELS[name]}")
        prediction = predict_features(
            model, test_features, test_indices, config.batch_size, device
        )
        arrays[f"prediction_{name}"] = prediction
        accuracy[name] = regression_summary(target, prediction)

    events = reconstruct_split_events(config.data_dir)
    rng = np.random.default_rng(config.seed + 30)
    pairs = distinct_rows(len(events), config.metric_samples, 2, rng)
    triplets = distinct_rows(len(events), config.metric_samples, 3, rng)
    identity = rng.choice(
        len(events), min(config.metric_samples, len(events)), replace=False
    )
    arrays.update(
        {
            "metric_pairs": pairs,
            "metric_triplets": triplets,
            "metric_identity": identity,
        }
    )

    metric_summary: dict[str, dict[str, float | int]] = {}
    for name, model in models.items():
        print(f"Benchmarking metric properties for {MODEL_LABELS[name]}")
        forward = predict_event_pairs(
            model, events, pairs[:, 0], pairs[:, 1], config.batch_size, device
        )
        reverse = predict_event_pairs(
            model, events, pairs[:, 1], pairs[:, 0], config.batch_size, device
        )
        self_distance = predict_event_pairs(
            model, events, identity, identity, config.batch_size, device
        )
        sides = []
        for first, second in ((0, 1), (1, 2), (0, 2)):
            sides.append(
                predict_event_pairs(
                    model,
                    events,
                    triplets[:, first],
                    triplets[:, second],
                    config.batch_size,
                    device,
                )
            )
        triangle_sides = np.stack(sides, axis=1)
        largest = np.max(triangle_sides, axis=1)
        triangle_residual = largest - (np.sum(triangle_sides, axis=1) - largest)
        symmetry_residual = np.abs(forward - reverse)

        arrays[f"pair_distance_{name}"] = forward
        arrays[f"self_distance_{name}"] = self_distance
        arrays[f"symmetry_residual_{name}"] = symmetry_residual
        arrays[f"triangle_residual_{name}"] = triangle_residual
        metric_summary[name] = {
            "nonnegative_pass_fraction": float(
                np.mean(forward >= -config.tolerance)
            ),
            "identity_pass_fraction": float(
                np.mean(np.abs(self_distance) <= config.tolerance)
            ),
            "symmetry_pass_fraction": float(
                np.mean(symmetry_residual <= config.tolerance)
            ),
            "triangle_pass_fraction": float(
                np.mean(triangle_residual <= config.tolerance)
            ),
            "negative_count": int(np.count_nonzero(forward < -config.tolerance)),
            "triangle_violation_count": int(
                np.count_nonzero(triangle_residual > config.tolerance)
            ),
            "max_abs_self_distance_gev": float(np.max(np.abs(self_distance))),
            "max_symmetry_residual_gev": float(np.max(symmetry_residual)),
            "max_triangle_violation_gev": float(max(0, np.max(triangle_residual))),
        }

    summary = {
        "accuracy": accuracy,
        "metric_properties": metric_summary,
        "benchmark": {
            "test_pair_count": len(test_indices),
            "source_test_event_count": len(events),
            "metric_pair_count": len(pairs),
            "metric_triplet_count": len(triplets),
            "identity_event_count": len(identity),
            "tolerance_gev": config.tolerance,
            "seed": config.seed,
        },
    }
    return summary, arrays


# %% [markdown]
# ## Plots and end-to-end workflow

# %%
def plot_training(histories: dict[str, dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, history in histories.items():
        epochs = np.arange(1, len(history["train_mape"]) + 1)
        color = MODEL_COLORS[name]
        axes[0].plot(
            epochs,
            history["train_mape"],
            color=color,
            marker="o",
            markersize=3,
            label=MODEL_LABELS[name],
        )
        axes[1].plot(
            epochs,
            history["val_mape"],
            color=color,
            marker="o",
            markersize=3,
            label=MODEL_LABELS[name],
        )
    for ax, title in zip(axes, ("Training", "Validation")):
        ax.set(xlabel="Epoch", ylabel="MAPE", title=title)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def plot_accuracy(summary: dict, arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    target = arrays["target"]
    predictions = [arrays[f"prediction_{name}"] for name in MODEL_LABELS]
    finite = np.concatenate([target, *predictions])
    low = float(min(0, np.quantile(finite, 0.002)))
    high = float(np.quantile(finite, 0.998))
    if high <= low:
        high = low + 1.0

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, name, prediction in zip(axes, MODEL_LABELS, predictions):
        ax.hist2d(
            target,
            prediction,
            bins=70,
            range=((low, high), (low, high)),
            norm=LogNorm(vmin=1),
            cmap="viridis",
            cmin=1,
        )
        ax.plot((low, high), (low, high), "--", color="black", linewidth=1)
        metrics = summary["accuracy"][name]
        ax.set(
            xlim=(low, high),
            ylim=(low, high),
            xlabel="Exact EMD [GeV]",
            ylabel="Predicted EMD [GeV]",
            title=MODEL_LABELS[name],
        )
        ax.text(
            0.04,
            0.96,
            f"MAPE = {metrics['mape']:.4f}\nMAE = {metrics['mae_gev']:.2f} GeV\n$R^2$ = {metrics['r_squared']:.4f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    figure.savefig(output_dir / "accuracy_benchmark.png", dpi=180)
    plt.close(figure)


def _histogram_limits(
    values: list[np.ndarray],
    full_low: bool = False,
    full_high: bool = False,
) -> tuple[float, float]:
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    low = np.min(finite) if full_low else np.quantile(finite, 0.002)
    high = np.max(finite) if full_high else np.quantile(finite, 0.998)
    low, high = min(float(low), 0.0), max(float(high), 0.0)
    if low == high:
        padding = max(abs(float(low)) * 0.1, 1e-6)
        low, high = low - padding, high + padding
    return float(low), float(high)


def plot_metric_benchmarks(
    summary: dict, arrays: dict[str, np.ndarray], output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    panels: list[tuple[str, str, str, Callable[[np.ndarray], np.ndarray]]] = [
        ("pair_distance", "Predicted distance [GeV]", "Non-negativity", lambda x: x),
        ("self_distance", "Predicted self-distance [GeV]", "Identity", lambda x: x),
        (
            "symmetry_residual",
            r"$|d(A,B)-d(B,A)|$ [GeV]",
            "Symmetry",
            lambda x: x,
        ),
        (
            "triangle_residual",
            "Largest side - sum of other sides [GeV]",
            "Triangle inequality",
            lambda x: x,
        ),
    ]
    pass_keys = (
        "nonnegative_pass_fraction",
        "identity_pass_fraction",
        "symmetry_pass_fraction",
        "triangle_pass_fraction",
    )
    # Always retain the complete forbidden side of non-negativity and triangle
    # inequality. Identity and symmetry use their complete observed tails.
    limit_options = (
        {"full_low": True},
        {"full_low": True, "full_high": True},
        {"full_high": True},
        {"full_high": True},
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (key, xlabel, title, transform), pass_key, limit_kwargs in zip(
        axes.flat, panels, pass_keys, limit_options
    ):
        values = [transform(arrays[f"{key}_{name}"]) for name in MODEL_LABELS]
        low, high = _histogram_limits(values, **limit_kwargs)
        edges = np.linspace(low, high, 61)
        if key in {"pair_distance", "triangle_residual"}:
            if key == "pair_distance" and low < 0:
                ax.axvspan(low, 0, color="#CC3311", alpha=0.08)
            if key == "triangle_residual" and high > 0:
                ax.axvspan(0, high, color="#CC3311", alpha=0.08)
        for name, value in zip(MODEL_LABELS, values):
            ax.hist(
                value,
                bins=edges,
                weights=np.full(len(value), 1 / len(value)),
                histtype="step",
                linewidth=1.8,
                color=MODEL_COLORS[name],
                label=(
                    f"{MODEL_LABELS[name]} "
                    f"(pass={summary['metric_properties'][name][pass_key]:.4f})"
                ),
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel=xlabel, ylabel="Fraction / bin", title=title, yscale="log")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=9)
    figure.savefig(output_dir / "metric_benchmarks.png", dpi=180)
    plt.close(figure)


def serializable_config(config: WorkflowConfig) -> dict:
    payload = asdict(config)
    payload["data_dir"] = str(config.data_dir.resolve())
    payload["output_dir"] = str(config.output_dir.resolve())
    return payload


def validate_config(config: WorkflowConfig) -> None:
    if config.stage not in {"all", "train", "benchmark"}:
        raise ValueError("stage must be all, train, or benchmark")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.metric_samples <= 0 or config.tolerance < 0:
        raise ValueError("metric_samples must be positive and tolerance non-negative")


def run_workflow(config: WorkflowConfig) -> dict:
    validate_config(config)
    config.data_dir = Path(config.data_dir)
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    print(f"Using {device}; data={config.data_dir}; output={config.output_dir}")

    models: dict[str, nn.Module] = {}
    histories: dict[str, dict] = {}
    if config.stage in {"all", "train"}:
        train_data = PairDataset(
            config.data_dir,
            "train",
            config.max_train_pairs,
            config.seed + 1,
            config.preload,
        )
        val_data = PairDataset(
            config.data_dir,
            "val",
            config.max_val_pairs,
            config.seed + 2,
            config.preload,
        )
        print(
            f"Selected {len(train_data):,} training and {len(val_data):,} validation pairs"
        )
        for name in MODEL_LABELS:
            models[name], histories[name] = train_one_model(
                name, config, train_data, val_data, device
            )
        if config.make_plots:
            plot_training(histories, config.output_dir)

    result: dict = {"config": serializable_config(config), "training": histories}
    if config.stage in {"all", "benchmark"}:
        if not models:
            models = {
                name: load_checkpoint(
                    config.output_dir / name / "best_model.pt", device
                )
                for name in MODEL_LABELS
            }
        benchmark_summary, arrays = benchmark_models(models, config, device)
        result.update(benchmark_summary)
        np.savez_compressed(config.output_dir / "benchmark_arrays.npz", **arrays)
        if config.make_plots:
            plot_accuracy(result, arrays, config.output_dir)
            plot_metric_benchmarks(result, arrays, config.output_dir)

    result["environment"] = {
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote results to {config.output_dir.resolve()}")
    return result


# %% [markdown]
# ## Command-line interface
#
# Run `python ma_pfn_demo.py --help` for every option. The final cell contains a
# deliberately short notebook configuration; increase its limits and epoch
# count for the release-scale result.

# %%
def parse_args() -> WorkflowConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--stage", choices=("all", "train", "benchmark"), default="all")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50, help="0 disables early stopping")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--phi-hidden-dim", type=int, default=100)
    parser.add_argument("--f-hidden-dim", type=int, default=100)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-val-pairs", type=int)
    parser.add_argument("--max-test-pairs", type=int, default=100_000)
    parser.add_argument("--metric-samples", type=int, default=20_000)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="copy selected train/validation arrays to float32 RAM for faster epochs",
    )
    parser.add_argument("--no-plots", action="store_true")
    values = vars(parser.parse_args())
    values["make_plots"] = not values.pop("no_plots")
    return WorkflowConfig(**values)


if __name__ == "__main__" and "get_ipython" not in globals():
    run_workflow(parse_args())


# %%
# Notebook users: edit these values, then run all cells.
if "get_ipython" in globals():
    notebook_config = WorkflowConfig(
        data_dir=Path("data"),
        output_dir=Path("results_notebook"),
        epochs=5,
        patience=0,
        max_train_pairs=50_000,
        max_val_pairs=10_000,
        max_test_pairs=20_000,
        metric_samples=2_000,
    )
    notebook_results = run_workflow(notebook_config)
