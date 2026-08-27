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
# This notebook trains a metric-aware particle flow network (MA-PFN) and a
# stock PFN under identical conditions, then compares held-out EMD regression
# and four metric properties: non-negativity, identity, symmetry, and the
# triangle inequality.
#
# The important model and cache operations are written out in executable cells
# below. [`models.py`](models.py) contains the reusable versions used for
# training, and [`utils.py`](utils.py) holds the less interesting data-loader,
# training-loop, plotting, and command-line machinery.

# %%
"""Train and benchmark a minimal MA-PFN and stock-PFN comparison."""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F

from models import MAPFN, PFN
from utils import (
    WorkflowConfig,
    benchmark_inference_throughput,
    load_checkpoint,
    prepare_demo_data,
    reconstruct_split_events,
    run_workflow,
    select_device,
)

from IPython.display import Image as NotebookImage, display


# %% [markdown]
# ## What changes in MA-PFN?
#
# Both networks use the same kind of particle-level encoder. The stock PFN
# feeds the event-identity tag into that encoder, sums every particle embedding,
# and applies an unconstrained regression head. MA-PFN makes four changes:
#
# 1. the `-1`/`+1` tag selects a pool but is zeroed before particle encoding;
# 2. one joint head receives the latent sum and the *signed* difference;
# 3. the head is evaluated in both event orientations and those outputs are
#    averaged; and
# 4. a softplus scale is multiplied by the mean absolute latent separation.
#
# Together these changes enforce non-negativity, identity, and symmetry by
# construction. At the release dimensions below, MA-PFN has 50,265 trainable
# parameters and the matched stock PFN has 43,865. The selected production
# checkpoint is epoch 648 of the extended run (698 epochs completed) and has
# SHA-256 `05a16361bbc8a8c0c137f530f1fac3c7d9990d2145fdb62e54c2f0b422d5551d`;
# the trained weights are not bundled with this tutorial.

# %% [markdown]
# ## Configure the demonstration
#
# Parameter choices are exposed below so a user can change them in one
# place. These defaults are deliberately small for a tutorial.
# Comments identify release-scale values where they differ.

# %%
config = WorkflowConfig(
    # Data and workflow
    data_dir=Path("data"),
    output_dir=Path("results_notebook"),
    device="auto",  # automatically use CUDA when available
    cpu_threads=4,  # avoids thread-launch overhead for this small CPU workload
    seed=23_411,    # selected production run

    # Architecture (shared wherever possible for a controlled comparison)
    latent_dim=64,  # release scale: 64
    phi_hidden_dim=100,  # release scale: 100
    f_hidden_dim=100,  # release scale: 100

    # Optimization
    epochs=10,           # release scale: 700
    patience=0,          # release scale: 50; 0 disables early stopping
    batch_size=1_024,
    learning_rate=1e-3,  # release scale: 1e-4
    weight_decay=0.0,    # matched production setting
    loss="hybrid",
    mae_weight=0.25,     # matched production setting
    mae_scale=90.0,      # GeV; makes the MAE contribution dimensionless
    num_workers=0,
    preload=False,       # True is faster if the selected arrays fit in host RAM

    # Reproducible demo subsets and final benchmarks
    max_train_pairs=100_000,
    max_val_pairs=None,
    max_test_pairs=None,   # paper benchmark: all 798,216 held-out pairs
    metric_samples=1_024,  # paper metric-property study: 1,000,000
    tolerance=1e-3,        # GeV; pass/fail tolerance for metric properties
    cache_inference=True,  # reuse event embeddings, as in the timing benchmark
    make_plots=True,
)

# Two banks of this many events make events_per_bank**2 throughput pairs.
throughput_events_per_bank = 32
throughput_repetitions = 3


# %% [markdown]
# ## Find the full data—or extract its tutorial subset
#
# If all six release arrays exist in `data/`, the notebook uses them.
# Otherwise, it extracts `ma_pfn_tutorial.npz` under
# `results_notebook/tutorial_data/`. The archive selects 448/64/64 real
# source events from the released event-disjoint train/validation/test splits,
# retains every unordered pair among them, and copies the corresponding exact
# EMD targets. 
#
# The training split contains 100,128 complete pairs; the configuration above
# selects exactly 100,000 of them reproducibly for the tutorial run.

# %%
data_info = prepare_demo_data(
    requested_dir=config.data_dir,
    fallback_dir=config.output_dir / "tutorial_data",
)
config.data_dir = Path(data_info["data_dir"])
print(f"Using {data_info['kind']} data from {config.data_dir}")
display({
    "kind": data_info["kind"],
    "archive_source": data_info.get("archive_source"),
    "data_dir": data_info["data_dir"],
    "zenodo_record_id": data_info.get("source", {}).get("zenodo_record_id"),
    "splits": {
        split: {
            "source_events": details["selected_event_count"],
            "pairs": details["pair_count"],
            "feature_shape": details["feature_shape"],
            "target_range_gev": (
                details["target_min_gev"],
                details["target_max_gev"],
            ),
        }
        for split, details in data_info.get("splits", {}).items()
    },
})


# %% [markdown]
# ## The architecture change, in code
#
# These two functions spell out the important operations in the stock PFN and
# MA-PFN forward passes. The reusable model classes in `models.py` organize the
# same operations for training.

# %%
def stock_pfn_forward(model: PFN, tagged_pairs: torch.Tensor) -> torch.Tensor:
    """Stock PFN: encode tagged particles, sum once, use a free regression head."""

    valid_particle = (tagged_pairs[..., :3].abs().sum(dim=-1) > 0).unsqueeze(-1)
    phi = F.relu(model.phi_fc1(tagged_pairs))
    phi = F.relu(model.phi_fc2(phi))
    phi = model.phi_fc3(phi) * valid_particle
    pooled_pair = phi.sum(dim=1)

    prediction = F.relu(model.f_fc1(pooled_pair))
    prediction = F.relu(model.f_fc2(prediction))
    prediction = F.relu(model.f_fc3(prediction))
    return model.f_fc4(prediction)[:, 0]


def metric_aware_forward(model: MAPFN, tagged_pairs: torch.Tensor) -> torch.Tensor:
    """MA-PFN: pool events separately and constrain how their latents interact."""

    valid_particle = (tagged_pairs[..., :3].abs().sum(dim=-1) > 0).unsqueeze(-1)
    event_id = tagged_pairs[..., 3:4]
    kinematics = tagged_pairs.clone()
    kinematics[..., 3] = 0.0  # tag chooses a pool; the encoder cannot learn it

    phi = F.relu(model.phi_fc1(kinematics))
    phi = F.relu(model.phi_fc2(phi))
    phi = model.phi_fc3(phi) * valid_particle
    first = (phi * (event_id == -1)).sum(dim=1)
    second = (phi * (event_id == 1)).sum(dim=1)

    latent_sum = first + second
    latent_difference = first - second

    def joint_log_scale(difference: torch.Tensor) -> torch.Tensor:
        features = torch.cat((latent_sum, difference), dim=1)
        features = F.relu(model.f_joint1(features))
        features = F.relu(model.f_joint2(features))
        features = F.relu(model.f_joint3(features))
        return model.f_joint4(features)[:, 0]

    # Swapping the events negates only the signed difference. Averaging the two
    # orientations therefore makes the learned scale exactly symmetric.
    symmetric_log_scale = 0.5 * (
        joint_log_scale(latent_difference)
        + joint_log_scale(-latent_difference)
    )
    latent_separation = torch.mean(torch.abs(latent_difference), dim=1)
    prediction = latent_separation * F.softplus(symmetric_log_scale)

    identical = torch.isclose(first, second, rtol=1e-5, atol=1e-6).all(dim=1)
    return torch.where(identical, torch.zeros_like(prediction), prediction)


# %% [markdown]
# ## The constructed training loss, in code
#
# Pure MAPE treats a fixed error as increasingly important as the target gets
# smaller; pure MAE instead emphasizes large absolute misses. The constructed
# loss retains both signals:
#
# $$
# \mathcal{L}_{\mathrm{hybrid}}
# = \operatorname{MAPE}
# + \lambda\,\frac{\operatorname{MAE}}{s},
# \qquad
# \operatorname{MAPE}=\frac{1}{n}\sum_i
# \frac{|y_i-\hat y_i|}{|y_i|+\epsilon}.
# $$
#
# Dividing MAE by the fixed energy scale $s$ makes that term dimensionless.
# The tutorial uses the matched production values $\lambda=0.25$ and
# $s=90\ \mathrm{GeV}$. Training, checkpoint selection, and early stopping use
# the combined objective, while MAPE and MAE remain visible diagnostics.

# %%
def hybrid_emd_loss_written_out(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mae_weight: float,
    mae_scale: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (hybrid objective, MAPE, MAE) using the displayed equation."""

    absolute_error = torch.abs(target - prediction)
    mape = torch.mean(absolute_error / (target.abs() + eps))
    mae = torch.mean(absolute_error)
    objective = mape + mae_weight * mae / mae_scale
    return objective, mape, mae


# %%
sample_target = torch.from_numpy(
    np.array(
        np.load(config.data_dir / "train_targets.npy", mmap_mode="r")[:4],
        dtype=np.float32,
    )
)
sample_prediction = sample_target * torch.tensor([0.8, 1.2, 0.9, 1.1])
objective, mape, mae = hybrid_emd_loss_written_out(
    sample_prediction,
    sample_target,
    config.mae_weight,
    config.mae_scale,
)
display({
    "objective_used_for_training": (
        f"MAPE + {config.mae_weight:g} * MAE / {config.mae_scale:g} GeV"
    ),
    "example_hybrid_objective": float(objective),
    "example_mape": float(mape),
    "example_mae_gev": float(mae),
})


# %% [markdown]
# ## Cached inference
#
# A data set with `N` unique events contains up to `N(N-1)/2` pairs. Re-running
# the particle encoder for each pair repeats the expensive part roughly `N`
# times. With `cache_inference=True`, inference instead:
#
# 1. reconstructs the unique held-out events;
# 2. transfers and encodes each event with `model.encode_events` before evaluating
#    pairs;
# 3. retains the event latents on the device; and
# 4. gathers two cached latents per requested pair and evaluate only the pair
#    head with `model.pairwise_from_latents`.
#
# MA-PFN needs one shared latent per event because its encoder does not learn the
# event tag. The stock PFN encoder does learn that tag, so it caches each event
# twice: once for its `-1` (first-event) role and once for its `+1`
# (second-event) role. Set the option to `False` to exercise the slower legacy
# path that rebuilds and re-encodes every tagged pair. The explicit cache cell
# below performs these operations after training.


# %% [markdown]
# ## Train, benchmark, and plot
#
# The workflow trains both models with the displayed hybrid loss, selects their
# best checkpoints by validation hybrid objective, evaluates the held-out test
# pairs, runs the metric-property benchmarks, and writes three plots plus
# machine-readable JSON/NPZ results.

# %%
results = run_workflow(config)


# %% [markdown]
# ## Training curves

# %%
display(NotebookImage(filename=str(config.output_dir / "training_curves.png")))


# %% [markdown]
# ## Held-out EMD regression

# %%
display(NotebookImage(filename=str(config.output_dir / "accuracy_benchmark.png")))


# %% [markdown]
# ## Metric-property benchmarks
#
# Each panel shows the full distribution relevant to one metric property. The
# dashed line is the boundary of the allowed region, and the annotation reports
# the fraction passing at the configured tolerance.

# %%
display(NotebookImage(filename=str(config.output_dir / "metric_benchmarks.png")))


# %% [markdown]
# ## Build and use the latent cache explicitly
#
# MA-PFN encodes every unique event once. The stock PFN has learned the event
# tag, so the same event has one latent for the first (`-1`) role and another
# for the second (`+1`) role. Pair inference is then just two indexed gathers
# followed by the small regression head.

# %%
def cache_events_explicitly(
    model: torch.nn.Module,
    event_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    with torch.inference_mode():
        if isinstance(model, MAPFN):
            shared = model.encode_events(event_tensor)
            return shared, shared, 1
        first_role = model.encode_events(event_tensor, event_id=-1.0)
        second_role = model.encode_events(event_tensor, event_id=1.0)
        return first_role, second_role, 2


def score_cached_pairs_explicitly(
    model: torch.nn.Module,
    cache: tuple[torch.Tensor, torch.Tensor, int],
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
) -> torch.Tensor:
    first_role, second_role, _ = cache
    with torch.inference_mode():
        return model.pairwise_from_latents(
            first_role.index_select(0, first_indices),
            second_role.index_select(0, second_indices),
        )


# %%
device = select_device(config.device)
held_out_events = reconstruct_split_events(config.data_dir)
event_tensor = torch.from_numpy(np.ascontiguousarray(held_out_events)).to(device)
first_indices = torch.tensor([0, 0, 1, 2, 3, 5, 8, 13], device=device)
second_indices = torch.tensor([1, 2, 3, 4, 5, 8, 13, 21], device=device)

cache_summary = {}
for name, label in (("ma_pfn", "MA-PFN"), ("pfn", "PFN")):
    model = load_checkpoint(config.output_dir / name / "best_model.pt", device)
    cache = cache_events_explicitly(model, event_tensor)
    cached = score_cached_pairs_explicitly(
        model, cache, first_indices, second_indices
    ).cpu().numpy()
    cache_summary[label] = {
        "unique_events": len(held_out_events),
        "latent_shape": list(cache[0].shape),
        "event_encoding_passes": cache[2],
        "example_predictions_gev": cached.tolist(),
    }
display(cache_summary)


# %% [markdown]
# ## Cached versus pair-encoding throughput
#
# This fixed-workload check uses the same event pairs and batch size for both
# paths. **Pair encode** gathers resident events, rebuilds the tagged pair
# tensor, and runs the particle encoder for every pair. **Cached cold** includes
# the one-time event/index transfer and event encoding; **cached resident**
# times repeated inference after those tensors are already on the selected
# device. Both paths return predictions to host memory, and the helper verifies
# that their outputs agree before reporting throughput.
#
# This is an illustrative comparison, not a replacement for the paper's
# controlled timing suite. Increase `throughput_events_per_bank` for a longer,
# more stable measurement.

# %%
throughput = benchmark_inference_throughput(
    config,
    events_per_bank=throughput_events_per_bank,
    repetitions=throughput_repetitions,
    verbose=False,
)

labels = [metrics["label"] for metrics in throughput["models"].values()]
x = np.arange(len(labels))
width = 0.25

figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
rate_series = (
    ("Pair encoding", "pair_encoding_resident_pairs_per_second"),
    ("Cached, cold", "cached_cold_pairs_per_second"),
    ("Cached, resident", "cached_resident_pairs_per_second"),
)
for offset, (label, key) in zip((-width, 0, width), rate_series):
    values = [metrics[key] for metrics in throughput["models"].values()]
    axes[0].bar(x + offset, values, width, label=label)
axes[0].set(
    xticks=x,
    xticklabels=labels,
    ylabel="Pairs / second",
    title="Inference throughput",
    yscale="log",
)
axes[0].legend(frameon=False, fontsize=8)

speedup_series = (
    ("Cold", "cached_cold_speedup"),
    ("Resident", "cached_resident_speedup"),
)
for offset, (label, key) in zip((-width / 2, width / 2), speedup_series):
    values = [metrics[key] for metrics in throughput["models"].values()]
    bars = axes[1].bar(x + offset, values, width, label=label)
    axes[1].bar_label(bars, fmt="%.1fx", padding=3, fontsize=8)
axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
axes[1].set(
    xticks=x,
    xticklabels=labels,
    ylabel="Speedup over pair encoding",
    title="Benefit of caching",
)
axes[1].legend(frameon=False, fontsize=8)

for axis in axes:
    axis.grid(axis="y", alpha=0.2)
figure.savefig(config.output_dir / "inference_throughput.png", dpi=180)
plt.show()


# %% [markdown]
# The figures are also saved in `results_notebook/` as `training_curves.png`,
# `accuracy_benchmark.png`, `metric_benchmarks.png`, and
# `inference_throughput.png`.
