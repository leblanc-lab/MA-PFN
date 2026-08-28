"""Data, training, benchmark, and plotting helpers for the MA-PFN demo."""

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
from torch.utils.data import DataLoader, Dataset

from models import HybridEMDLoss, MAPFN, PFN


MODEL_LABELS = {"ma_pfn": "MA-PFN", "pfn": "PFN"}
MODEL_COLORS = {"ma_pfn": "#0072B2", "pfn": "#D55E00"}
MODEL_ARCHITECTURES = {"ma_pfn": "joint", "pfn": "baseline"}
DATA_SPLITS = ("train", "val", "test")
TUTORIAL_SUBSET_FILENAME = "ma_pfn_tutorial.npz"


@dataclass
class WorkflowConfig:
    """All settings needed to reproduce a training and benchmark run."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("results")
    stage: str = "all"
    epochs: int = 700
    patience: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    loss: str = "hybrid"
    mae_weight: float = 0.25
    mae_scale: float = 90.0
    num_workers: int = 0
    seed: int = 23411
    latent_dim: int = 64
    phi_hidden_dim: int = 100
    f_hidden_dim: int = 100
    max_train_pairs: int | None = None
    max_val_pairs: int | None = None
    max_test_pairs: int | None = 100_000
    metric_samples: int = 20_000
    tolerance: float = 1e-3
    device: str = "auto"
    cpu_threads: int | None = None
    preload: bool = False
    cache_inference: bool = True
    make_plots: bool = True


def make_model(name: str, config: WorkflowConfig) -> nn.Module:
    """Build either model with the shared architecture settings."""

    if name not in MODEL_LABELS:
        raise ValueError(f"Unknown model name: {name}")
    model_class: type[nn.Module] = MAPFN if name == "ma_pfn" else PFN
    return model_class(
        input_dim=4,
        latent_dim=config.latent_dim,
        phi_hidden_dim=config.phi_hidden_dim,
        f_hidden_dim=config.f_hidden_dim,
    )


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped event-pair features and exact EMD targets."""

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
            missing = [
                str(path)
                for path in (self.features_path, self.targets_path)
                if not path.is_file()
            ]
            raise FileNotFoundError(
                "Missing pair data: "
                + ", ".join(missing)
                + ". Download all six release arrays described in README.md, or "
                "use prepare_demo_data(...) to extract the real tutorial subset."
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
        if limit is not None and 0 < limit < len(self.targets):
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


def evaluate_loss_components(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: HybridEMDLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"objective": 0.0, "mape": 0.0, "mae": 0.0}
    examples = 0
    with torch.inference_mode():
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            components = loss_fn.components(model(features), target)
            for key, value in zip(totals, components):
                totals[key] += value.item() * len(features)
            examples += len(features)
    return {key: total / examples for key, total in totals.items()}


def train_one_model(
    name: str,
    config: WorkflowConfig,
    train_data: PairDataset,
    val_data: PairDataset,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    set_seed(config.seed)
    model = make_model(name, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = HybridEMDLoss(
        mae_weight=config.mae_weight,
        mae_scale=config.mae_scale,
        loss=config.loss,
    )

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
    print(f"  objective: {loss_fn.description}", flush=True)
    print(f"  AdamW weight decay: {config.weight_decay:g}", flush=True)

    history: dict[str, object] = {
        "train_objective": [],
        "train_mape": [],
        "train_mae": [],
        "val_objective": [],
        "val_mape": [],
        "val_mae": [],
    }
    best_loss = float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    started = time.time()
    for epoch in range(config.epochs):
        model.train()
        totals = {"objective": 0.0, "mape": 0.0, "mae": 0.0}
        examples = 0
        for features, target in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            components = loss_fn.components(model(features), target)
            objective = components[0]
            objective.backward()
            optimizer.step()
            for key, value in zip(totals, components):
                totals[key] += value.item() * len(features)
            examples += len(features)

        train_metrics = {key: total / examples for key, total in totals.items()}
        val_metrics = evaluate_loss_components(model, val_loader, loss_fn, device)
        for key in totals:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])
        improved = val_metrics["objective"] < best_loss
        if improved:
            best_loss = val_metrics["objective"]
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            checkpoint = {
                "format_version": 2,
                "architecture": MODEL_ARCHITECTURES[name],
                "model_name": name,
                "loss": {
                    "name": config.loss,
                    "mae_weight": config.mae_weight,
                    "mae_scale_gev": config.mae_scale,
                    "description": loss_fn.description,
                },
                "model_kwargs": {
                    "input_dim": 4,
                    "latent_dim": config.latent_dim,
                    "phi_hidden_dim": config.phi_hidden_dim,
                    "f_hidden_dim": config.f_hidden_dim,
                },
                "training": {
                    "seed": config.seed,
                    "epoch": epoch + 1,
                    "optimizer": "AdamW",
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "batch_size": config.batch_size,
                },
                "model_state_dict": model.state_dict(),
            }
            torch.save(checkpoint, run_dir / "best_model.pt")

        elapsed = (time.time() - started) / 60
        marker = " *" if improved else ""
        print(
            f"  epoch {epoch + 1:03d}/{config.epochs}: "
            f"train objective={train_metrics['objective']:.6f}, "
            f"val objective={val_metrics['objective']:.6f}, "
            f"val MAPE={val_metrics['mape']:.6f}, "
            f"val MAE={val_metrics['mae']:.4f} GeV, "
            f"{elapsed:.1f} min{marker}",
            flush=True,
        )
        if config.patience and epoch - best_epoch >= config.patience:
            print(f"  early stopping after epoch {epoch + 1}", flush=True)
            break

    history.update(
        {
            "model": name,
            "parameter_count": parameter_count,
            "loss": config.loss,
            "objective_description": loss_fn.description,
            "mae_weight": config.mae_weight,
            "mae_scale_gev": config.mae_scale,
            "optimizer": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "epochs_completed": len(history["train_objective"]),
            "best_epoch": best_epoch + 1,
            "best_val_objective": best_metrics["objective"],
            "best_val_mape": best_metrics["mape"],
            "best_val_mae": best_metrics["mae"],
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
    architecture = checkpoint["architecture"]
    if architecture == "joint":
        model_class: type[nn.Module] = MAPFN
    elif architecture in {"baseline", "pfn"}:
        model_class = PFN
    elif architecture == "ma_pfn":
        raise ValueError(
            "This is a legacy factorized MA-PFN checkpoint and is incompatible "
            "with the selected symmetric joint architecture"
        )
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model = model_class(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def subset_indices(length: int, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit <= 0 or limit >= length:
        return np.arange(length)
    return np.sort(np.random.default_rng(seed).choice(length, limit, replace=False))


def expected_data_paths(data_dir: Path) -> tuple[Path, ...]:
    """Return the six NumPy files consumed by the workflow."""

    return tuple(
        data_dir / f"{split}_{kind}.npy"
        for split in DATA_SPLITS
        for kind in ("features", "targets")
    )


def _validated_subset_metadata(archive: np.lib.npyio.NpzFile) -> dict:
    expected_keys = {
        f"{split}_{kind}" for split in DATA_SPLITS for kind in ("features", "targets")
    }
    missing = expected_keys.difference(archive.files)
    if missing or "metadata_json" not in archive.files:
        raise ValueError(
            "Tutorial subset archive is missing: " + ", ".join(sorted(missing))
        )
    metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("format") != "ma-pfn-real-tutorial-subset":
        raise ValueError("Unrecognized tutorial subset format")
    for split in DATA_SPLITS:
        features = archive[f"{split}_features"]
        targets = archive[f"{split}_targets"]
        if features.ndim != 3 or features.shape[-1] != 4:
            raise ValueError(f"Invalid {split} feature shape in tutorial subset")
        if targets.ndim != 1 or len(features) != len(targets):
            raise ValueError(f"Mismatched {split} arrays in tutorial subset")
    return metadata


def _extract_subset(archive_path: Path, data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = _validated_subset_metadata(archive)
        for split in DATA_SPLITS:
            for kind in ("features", "targets"):
                np.save(data_dir / f"{split}_{kind}.npy", archive[f"{split}_{kind}"])
    (data_dir / "tutorial_subset.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def prepare_demo_data(
    requested_dir: Path,
    fallback_dir: Path,
    subset_archive: Path | None = None,
) -> dict:
    """Use complete supplied arrays or extract the real release-data subset.

    A partially populated supplied directory is treated as an error because
    silently combining different releases would invalidate the split contract.
    The compact archive retains all pairs among selected source events from each
    original event-disjoint split and copies the corresponding exact EMD targets.
    """

    requested_dir = Path(requested_dir)
    fallback_dir = Path(fallback_dir)
    requested_paths = expected_data_paths(requested_dir)
    present = [path.is_file() for path in requested_paths]
    if all(present):
        return {
            "kind": "full_release",
            "extracted": False,
            "data_dir": str(requested_dir),
            "description": "supplied train/validation/test NumPy arrays",
        }
    if any(present):
        missing = [str(path) for path, exists in zip(requested_paths, present) if not exists]
        raise FileNotFoundError(
            "The supplied data directory is incomplete; missing " + ", ".join(missing)
        )

    fallback_paths = expected_data_paths(fallback_dir)
    metadata_path = fallback_dir / "tutorial_subset.json"
    reusable = all(path.is_file() for path in fallback_paths) and metadata_path.is_file()
    if reusable:
        metadata = json.loads(metadata_path.read_text())
        reusable = (
            metadata.get("format") == "ma-pfn-real-tutorial-subset"
            and metadata.get("version") == 4
        )
    if not reusable:
        archive_path = (
            Path(subset_archive)
            if subset_archive is not None
            else Path(__file__).resolve().with_name(TUTORIAL_SUBSET_FILENAME)
        )
        archive_source = "bundled"
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Tutorial subset archive not found: {archive_path}"
            )
        metadata = _extract_subset(archive_path, fallback_dir)
    else:
        archive_source = "extracted_cache"

    return {
        "kind": "real_tutorial_subset",
        "extracted": not reusable,
        "archive_source": archive_source,
        "data_dir": str(fallback_dir),
        "description": metadata["selection"],
        "splits": metadata["splits"],
    }


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
        raise ValueError(f"{path} is not a complete combinations-ordered pair split")
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


@dataclass(frozen=True)
class EventLatentCache:
    """Resident event latents used by the timing-benchmark inference path."""

    first_role: torch.Tensor
    second_role: torch.Tensor
    encoding_passes: int


@dataclass(frozen=True)
class ResidentPairIndices:
    """Pair indices transferred once and retained beside cached latents."""

    first: torch.Tensor
    second: torch.Tensor


def build_event_latent_cache(
    model: nn.Module, events: np.ndarray, device: torch.device
) -> EventLatentCache:
    """Transfer and encode each unique event before evaluating any pairs.

    MA-PFN does not expose the event tag to its particle encoder, so one shared
    latent bank serves both pair roles. The stock PFN learns the tag, requiring
    one cached bank for the ``-1`` role and one for the ``+1`` role.
    """

    event_tensor = torch.from_numpy(
        np.ascontiguousarray(events, dtype=np.float32)
    ).to(device)
    with torch.inference_mode():
        if isinstance(model, PFN):
            first = model.encode_events(event_tensor, event_id=-1.0)
            second = model.encode_events(event_tensor, event_id=1.0)
            return EventLatentCache(first, second, encoding_passes=2)
        if not isinstance(model, MAPFN):
            raise TypeError(
                f"Cached inference is unsupported for {type(model).__name__}"
            )
        shared = model.encode_events(event_tensor)
        return EventLatentCache(shared, shared, encoding_passes=1)


def build_banked_event_latent_cache(
    model: nn.Module,
    first_events: np.ndarray,
    second_events: np.ndarray,
    device: torch.device,
) -> EventLatentCache:
    """Encode two event banks separately, matching the timing-study setup."""

    first_tensor = torch.from_numpy(
        np.ascontiguousarray(first_events, dtype=np.float32)
    ).to(device)
    second_tensor = torch.from_numpy(
        np.ascontiguousarray(second_events, dtype=np.float32)
    ).to(device)
    return _encode_banked_event_tensors(model, first_tensor, second_tensor)


def _encode_banked_event_tensors(
    model: nn.Module,
    first_events: torch.Tensor,
    second_events: torch.Tensor,
) -> EventLatentCache:
    with torch.inference_mode():
        if isinstance(model, PFN):
            first = model.encode_events(first_events, event_id=-1.0)
            second = model.encode_events(second_events, event_id=1.0)
        elif isinstance(model, MAPFN):
            first = model.encode_events(first_events)
            second = model.encode_events(second_events)
        else:
            raise TypeError(
                f"Cached inference is unsupported for {type(model).__name__}"
            )
    return EventLatentCache(first, second, encoding_passes=2)


def build_resident_pair_indices(
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    device: torch.device,
) -> ResidentPairIndices:
    if len(first_indices) != len(second_indices):
        raise ValueError("First and second pair-index arrays must have equal length")
    first = torch.from_numpy(
        np.ascontiguousarray(first_indices, dtype=np.int64)
    ).to(device)
    second = torch.from_numpy(
        np.ascontiguousarray(second_indices, dtype=np.int64)
    ).to(device)
    return ResidentPairIndices(first, second)


def predict_resident_cached_pairs(
    model: nn.Module,
    cache: EventLatentCache,
    indices: ResidentPairIndices,
    batch_size: int,
) -> np.ndarray:
    """Evaluate cached latents with pair indices already resident on-device."""

    pair_count = indices.first.numel()
    prediction = np.empty(pair_count, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, pair_count, batch_size):
            stop = min(start + batch_size, pair_count)
            first = cache.first_role.index_select(0, indices.first[start:stop])
            second = cache.second_role.index_select(0, indices.second[start:stop])
            prediction[start:stop] = (
                model.pairwise_from_latents(first, second).float().cpu().numpy()
            )
    return prediction


def _predict_resident_encoded_pairs(
    model: nn.Module,
    first_events: torch.Tensor,
    second_events: torch.Tensor,
    indices: ResidentPairIndices,
    batch_size: int,
) -> np.ndarray:
    """Run the full particle encoder with events and indices already resident."""

    pair_count = indices.first.numel()
    prediction = np.empty(pair_count, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, pair_count, batch_size):
            stop = min(start + batch_size, pair_count)
            first = first_events.index_select(0, indices.first[start:stop])
            second = second_events.index_select(0, indices.second[start:stop])
            first = nn.functional.pad(first, (0, 1), value=-1.0)
            second = nn.functional.pad(second, (0, 1), value=1.0)
            pair = torch.cat((first, second), dim=1)
            prediction[start:stop] = model(pair).float().cpu().numpy()
    return prediction


def predict_cached_pairs(
    model: nn.Module,
    cache: EventLatentCache,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Gather resident latents and evaluate only the pairwise regression head."""

    indices = build_resident_pair_indices(first_indices, second_indices, device)
    return predict_resident_cached_pairs(model, cache, indices, batch_size)


def combination_pair_indices(
    event_count: int, selected_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map rows in combinations order back to their two source-event indices."""

    first, second = np.triu_indices(event_count, k=1)
    return first[selected_rows], second[selected_rows]


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
    test_targets = np.load(config.data_dir / "test_targets.npy", mmap_mode="r")
    test_indices = subset_indices(
        len(test_targets), config.max_test_pairs, config.seed + 20
    )
    target = np.asarray(test_targets[test_indices], dtype=np.float32)
    events = reconstruct_split_events(config.data_dir)

    test_features: np.ndarray | None = None
    test_first: np.ndarray | None = None
    test_second: np.ndarray | None = None
    if config.cache_inference:
        test_first, test_second = combination_pair_indices(len(events), test_indices)
    else:
        test_features = np.load(
            config.data_dir / "test_features.npy", mmap_mode="r"
        )

    arrays: dict[str, np.ndarray] = {
        "test_indices": test_indices,
        "target": target,
    }
    accuracy: dict[str, dict[str, float]] = {}
    latent_caches: dict[str, EventLatentCache] = {}
    for name, model in models.items():
        if config.cache_inference:
            print(
                f"Caching {len(events):,} unique test-event latents for "
                f"{MODEL_LABELS[name]}"
            )
            cache = build_event_latent_cache(model, events, device)
            latent_caches[name] = cache
            assert test_first is not None and test_second is not None
            prediction = predict_cached_pairs(
                model,
                cache,
                test_first,
                test_second,
                config.batch_size,
                device,
            )
        else:
            print(
                f"Predicting {len(test_indices):,} held-out paired tensors with "
                f"{MODEL_LABELS[name]}"
            )
            assert test_features is not None
            prediction = predict_features(
                model, test_features, test_indices, config.batch_size, device
            )
        arrays[f"prediction_{name}"] = prediction
        accuracy[name] = regression_summary(target, prediction)

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

        def predict(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            if config.cache_inference:
                return predict_cached_pairs(
                    model,
                    latent_caches[name],
                    first,
                    second,
                    config.batch_size,
                    device,
                )
            return predict_event_pairs(
                model, events, first, second, config.batch_size, device
            )

        forward = predict(pairs[:, 0], pairs[:, 1])
        reverse = predict(pairs[:, 1], pairs[:, 0])
        self_distance = predict(identity, identity)
        sides = []
        for first, second in ((0, 1), (1, 2), (0, 2)):
            sides.append(predict(triplets[:, first], triplets[:, second]))
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
            "inference_mode": (
                "cached_event_latents" if config.cache_inference else "paired_tensors"
            ),
            "cache_scope": "held_out_test_events" if config.cache_inference else None,
            "event_encoding_passes": {
                name: cache.encoding_passes for name, cache in latent_caches.items()
            },
        },
    }
    return summary, arrays


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference_throughput(
    config: WorkflowConfig,
    events_per_bank: int = 64,
    repetitions: int = 3,
    verbose: bool = True,
) -> dict:
    """Compare pair re-encoding with cold and resident-cache inference.

    The workload contains all cross pairs between two disjoint event banks.
    Event tensors and pair indices are transferred once for both paths, as in
    the timing study. Both paths return every prediction to host memory. The
    cached cold-start result includes that common transfer plus event encoding,
    while its resident result times only lookup and the pairwise head.
    """

    if events_per_bank <= 0 or repetitions <= 0:
        raise ValueError("events_per_bank and repetitions must be positive")

    device = select_device(config.device)
    events = reconstruct_split_events(Path(config.data_dir))
    required_events = 2 * events_per_bank
    if required_events > len(events):
        raise ValueError(
            f"Need {required_events} held-out events, but found only {len(events)}"
        )

    rng = np.random.default_rng(config.seed + 40)
    selected = rng.choice(len(events), required_events, replace=False)
    first_events = np.ascontiguousarray(
        events[selected[:events_per_bank]], dtype=np.float32
    )
    second_events = np.ascontiguousarray(
        events[selected[events_per_bank:]], dtype=np.float32
    )
    first_indices = np.repeat(
        np.arange(events_per_bank, dtype=np.int64), events_per_bank
    )
    second_indices = np.tile(
        np.arange(events_per_bank, dtype=np.int64), events_per_bank
    )
    pair_count = len(first_indices)
    warm_count = min(pair_count, config.batch_size)
    output: dict = {
        "configuration": {
            "device": str(device),
            "events_per_bank": events_per_bank,
            "unique_event_count": required_events,
            "pair_count": pair_count,
            "batch_size": config.batch_size,
            "repetitions": repetitions,
            "selected_source_event_indices": selected.tolist(),
            "workload": "all cross pairs between two disjoint event banks",
            "timing": "wall time; median after warm-up; predictions returned to host",
            "baseline": "resident events and indices; particle encoder run per pair",
        },
        "models": {},
    }

    for name, label in MODEL_LABELS.items():
        model = load_checkpoint(
            Path(config.output_dir) / name / "best_model.pt", device
        )

        _synchronize(device)
        start = time.perf_counter()
        first_tensor = torch.from_numpy(first_events).to(device)
        second_tensor = torch.from_numpy(second_events).to(device)
        resident_indices = build_resident_pair_indices(
            first_indices, second_indices, device
        )
        _synchronize(device)
        resident_setup_seconds = time.perf_counter() - start

        # Warm the full pair-encoding path once before collecting timings.
        warm_indices = ResidentPairIndices(
            resident_indices.first[:warm_count], resident_indices.second[:warm_count]
        )
        _predict_resident_encoded_pairs(
            model,
            first_tensor,
            second_tensor,
            warm_indices,
            config.batch_size,
        )
        paired_times = []
        paired_prediction = np.empty(0, dtype=np.float32)
        for _ in range(repetitions):
            _synchronize(device)
            start = time.perf_counter()
            paired_prediction = _predict_resident_encoded_pairs(
                model,
                first_tensor,
                second_tensor,
                resident_indices,
                config.batch_size,
            )
            _synchronize(device)
            paired_times.append(time.perf_counter() - start)

        _synchronize(device)
        start = time.perf_counter()
        cache = _encode_banked_event_tensors(model, first_tensor, second_tensor)
        _synchronize(device)
        cache_setup_seconds = time.perf_counter() - start

        predict_resident_cached_pairs(
            model, cache, warm_indices, config.batch_size
        )
        cached_times = []
        cached_prediction = np.empty(0, dtype=np.float32)
        for _ in range(repetitions):
            _synchronize(device)
            start = time.perf_counter()
            cached_prediction = predict_resident_cached_pairs(
                model, cache, resident_indices, config.batch_size
            )
            _synchronize(device)
            cached_times.append(time.perf_counter() - start)

        max_difference = float(
            np.max(np.abs(paired_prediction - cached_prediction), initial=0.0)
        )
        if not np.allclose(
            paired_prediction, cached_prediction, rtol=2e-5, atol=2e-5
        ):
            raise RuntimeError(
                f"Cached and re-encoded {label} predictions disagree "
                f"(max absolute difference {max_difference:.3g} GeV)"
            )

        paired_seconds = float(np.median(paired_times))
        cached_seconds = float(np.median(cached_times))
        paired_rate = pair_count / paired_seconds
        resident_rate = pair_count / cached_seconds
        paired_cold_rate = pair_count / (resident_setup_seconds + paired_seconds)
        cold_rate = pair_count / (
            resident_setup_seconds + cache_setup_seconds + cached_seconds
        )
        output["models"][name] = {
            "label": label,
            "event_encoding_passes": cache.encoding_passes,
            "resident_input_setup_seconds": resident_setup_seconds,
            "cache_encoding_seconds": cache_setup_seconds,
            "pair_encoding_seconds": paired_seconds,
            "cached_resident_seconds": cached_seconds,
            "pair_encoding_cold_pairs_per_second": paired_cold_rate,
            "pair_encoding_resident_pairs_per_second": paired_rate,
            "cached_cold_pairs_per_second": cold_rate,
            "cached_resident_pairs_per_second": resident_rate,
            "cached_cold_speedup": cold_rate / paired_cold_rate,
            "cached_resident_speedup": resident_rate / paired_rate,
            "max_abs_prediction_difference_gev": max_difference,
            "pair_encoding_repetition_seconds": paired_times,
            "cached_resident_repetition_seconds": cached_times,
        }

    if verbose:
        print(
            f"Inference throughput: {pair_count:,} pairs, "
            f"batch={config.batch_size:,}, device={device}"
        )
        print(
            f"{'Model':<8} {'pair encode':>13} {'cached cold':>13} "
            f"{'cached resident':>16} {'encode':>10} {'cold':>8} {'resident':>10}"
        )
        print(
            f"{'':<8} {'(pairs/s)':>13} {'(pairs/s)':>13} "
            f"{'(pairs/s)':>16} {'(ms)':>10} {'speedup':>8} {'speedup':>10}"
        )
        for metrics in output["models"].values():
            print(
                f"{metrics['label']:<8} "
                f"{metrics['pair_encoding_resident_pairs_per_second']:>10,.0f} p/s "
                f"{metrics['cached_cold_pairs_per_second']:>10,.0f} p/s "
                f"{metrics['cached_resident_pairs_per_second']:>13,.0f} p/s "
                f"{1_000 * metrics['cache_encoding_seconds']:>8.2f} "
                f"{metrics['cached_cold_speedup']:>7.2f}x "
                f"{metrics['cached_resident_speedup']:>9.2f}x"
            )
        print("Cold cached throughput includes one cache setup for this workload.")

    output_path = Path(config.output_dir) / "inference_throughput.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def plot_training(histories: dict[str, dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = (
        ("objective", "Hybrid objective"),
        ("mape", "MAPE"),
        ("mae", "MAE [GeV]"),
    )
    for name, history in histories.items():
        epochs = np.arange(1, len(history["train_objective"]) + 1)
        color = MODEL_COLORS[name]
        for axis, (key, ylabel) in zip(axes, panels):
            axis.plot(
                epochs,
                history[f"train_{key}"],
                color=color,
                linestyle=":",
                label=f"{MODEL_LABELS[name]} train",
            )
            axis.plot(
                epochs,
                history[f"val_{key}"],
                color=color,
                marker="o",
                markersize=3,
                label=f"{MODEL_LABELS[name]} validation",
            )
            axis.set(xlabel="Epoch", ylabel=ylabel)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
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
    for axis, name, prediction in zip(axes, MODEL_LABELS, predictions):
        axis.hist2d(
            target,
            prediction,
            bins=70,
            range=((low, high), (low, high)),
            norm=LogNorm(vmin=1),
            cmap="viridis",
            cmin=1,
        )
        axis.plot((low, high), (low, high), "--", color="black", linewidth=1)
        metrics = summary["accuracy"][name]
        axis.set(
            xlim=(low, high),
            ylim=(low, high),
            xlabel="Exact EMD [GeV]",
            ylabel="Predicted EMD [GeV]",
            title=MODEL_LABELS[name],
        )
        axis.text(
            0.04,
            0.96,
            f"MAPE = {metrics['mape']:.4f}\nMAE = {metrics['mae_gev']:.2f} GeV\n$R^2$ = {metrics['r_squared']:.4f}",
            transform=axis.transAxes,
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
    for axis, (key, xlabel, title, transform), pass_key, limit_kwargs in zip(
        axes.flat, panels, pass_keys, limit_options
    ):
        values = [transform(arrays[f"{key}_{name}"]) for name in MODEL_LABELS]
        low, high = _histogram_limits(values, **limit_kwargs)
        edges = np.linspace(low, high, 61)
        if key == "pair_distance" and low < 0:
            axis.axvspan(low, 0, color="#CC3311", alpha=0.08)
        if key == "triangle_residual" and high > 0:
            axis.axvspan(0, high, color="#CC3311", alpha=0.08)
        for name, value in zip(MODEL_LABELS, values):
            axis.hist(
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
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel=xlabel, ylabel="Fraction / bin", title=title, yscale="log")
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, fontsize=9)
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
    if config.loss not in {"mape", "mae", "hybrid"}:
        raise ValueError("loss must be mape, mae, or hybrid")
    if config.mae_weight < 0 or config.mae_scale <= 0:
        raise ValueError("mae_weight must be non-negative and mae_scale positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.metric_samples <= 0 or config.tolerance < 0:
        raise ValueError("metric_samples must be positive and tolerance non-negative")
    if config.cpu_threads is not None and config.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive when supplied")


def run_workflow(config: WorkflowConfig) -> dict:
    """Train both models, benchmark them, and write metrics and plots."""

    validate_config(config)
    config.data_dir = Path(config.data_dir)
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    if device.type == "cpu" and config.cpu_threads is not None:
        torch.set_num_threads(config.cpu_threads)
    thread_note = (
        f" ({torch.get_num_threads()} intra-op threads)" if device.type == "cpu" else ""
    )
    print(
        f"Using {device}{thread_note}; data={config.data_dir}; output={config.output_dir}"
    )

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
                name: load_checkpoint(config.output_dir / name / "best_model.pt", device)
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
        "torch_intraop_threads": torch.get_num_threads(),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote results to {config.output_dir.resolve()}")
    return result


def parse_args() -> WorkflowConfig:
    """Parse the command-line interface used by ``ma_pfn_demo.py``."""

    parser = argparse.ArgumentParser(
        description="Train and benchmark a minimal MA-PFN and stock-PFN comparison."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--stage", choices=("all", "train", "benchmark"), default="all")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument(
        "--patience", type=int, default=50, help="0 disables early stopping"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--loss", choices=("mape", "mae", "hybrid"), default="hybrid"
    )
    parser.add_argument("--mae-weight", type=float, default=0.25)
    parser.add_argument(
        "--mae-scale",
        type=float,
        default=90.0,
        help="GeV scale that makes the hybrid loss's MAE term dimensionless",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23411)
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
        "--cpu-threads",
        type=int,
        help="limit PyTorch intra-op threads (useful for small CPU demonstrations)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="copy selected train/validation arrays to float32 RAM for faster epochs",
    )
    parser.add_argument(
        "--no-cache-inference",
        action="store_true",
        help="rebuild tagged pair tensors instead of caching unique event latents",
    )
    parser.add_argument("--no-plots", action="store_true")
    values = vars(parser.parse_args())
    values["cache_inference"] = not values.pop("no_cache_inference")
    values["make_plots"] = not values.pop("no_plots")
    return WorkflowConfig(**values)


__all__ = [
    "EventLatentCache",
    "MODEL_LABELS",
    "PairDataset",
    "ResidentPairIndices",
    "TUTORIAL_SUBSET_FILENAME",
    "WorkflowConfig",
    "benchmark_inference_throughput",
    "benchmark_models",
    "build_banked_event_latent_cache",
    "build_event_latent_cache",
    "build_resident_pair_indices",
    "expected_data_paths",
    "load_checkpoint",
    "parse_args",
    "predict_cached_pairs",
    "predict_event_pairs",
    "predict_resident_cached_pairs",
    "prepare_demo_data",
    "reconstruct_split_events",
    "run_workflow",
    "select_device",
    "train_one_model",
]
