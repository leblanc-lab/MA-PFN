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
from torch.nn import functional as F

from models import HybridEMDLoss, MAPFN, PFN
from utils import (
    WorkflowConfig,
    benchmark_inference_throughput,
    load_checkpoint,
    parse_args,
    predict_event_pairs,
    prepare_demo_data,
    reconstruct_split_events,
    run_workflow,
    select_device,
)

if "get_ipython" in globals():
    from IPython.display import Image as NotebookImage, display


# %% [markdown]
# ## What changes in MA-PFN?
#
# Both networks use the same kind of particle-level encoder. The stock PFN
# feeds the event-identity tag into that encoder, sums every particle embedding,
# and applies an unconstrained regression head. MA-PFN makes four changes:
#
# 1. the `-1`/`+1` tag selects a pool but is zeroed before particle encoding;
# 2. the two event latents are combined only as `abs(A - B)` and `A + B`;
# 3. the difference branch has no biases, so identical inputs stay zero; and
# 4. the final product is absolute-valued.
#
# Together these changes enforce non-negativity, identity, and symmetry by
# construction.

# %% [markdown]
# ## Configure the demonstration
#
# Parameter choices are exposed below so a user can change them in one
# place. These defaults are deliberately small for a tutorial.
# Comments identify release-scale values where they differ. `stage` can be
# `"all"`, `"train"`, or `"benchmark"`.

# %%
config = WorkflowConfig(
    # Data and workflow
    data_dir=Path("data"),
    output_dir=Path("results_notebook"),
    stage="all",
    device="auto",  # automatically use CUDA when available
    cpu_threads=4,  # avoids thread-launch overhead for this small CPU workload
    seed=12_345,

    # Architecture (shared wherever possible for a controlled comparison)
    latent_dim=64,  # release scale: 64
    phi_hidden_dim=100,  # release scale: 100
    f_hidden_dim=100,  # release scale: 100

    # Optimization
    epochs=10,  # release scale: 500
    patience=0,  # release scale: 50; 0 disables early stopping
    batch_size=1_024,
    learning_rate=1e-3,  # release scale: 1e-4
    loss="hybrid",
    mae_weight=0.25,  # matched production setting
    mae_scale=90.0,  # GeV; makes the MAE contribution dimensionless
    num_workers=0,
    preload=False,  # True is faster if the selected arrays fit in host RAM

    # Reproducible demo subsets and final benchmarks
    max_train_pairs=100_000,
    max_val_pairs=None,
    max_test_pairs=None,  # release scale: 100_000
    metric_samples=1_024,  # release scale: 20_000
    tolerance=1e-3,  # GeV; pass/fail tolerance for metric properties
    cache_inference=True,  # reuse event embeddings, as in the timing benchmark
    make_plots=True,
)

# Two banks of this many events make events_per_bank**2 throughput pairs.
throughput_events_per_bank = 32
throughput_repetitions = 3
data_info = None
results = None  # dependent cells skip cleanly if preparation or training fails


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
if "get_ipython" in globals():
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
# These two functions spell out the exact forward passes used by the reusable
# classes. They deliberately use the named layers from `models.py`, so the cell
# below can check them numerically against `PFN.forward` and `MAPFN.forward`.
# The duplicated few lines are a readable specification, not an alternative
# implementation used for training.

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

    difference = torch.abs(first - second)  # unchanged by swapping the events
    total = first + second                  # unchanged by swapping the events
    difference = F.relu(model.f_diff1(difference))  # all f_diff layers are bias-free
    difference = F.relu(model.f_diff2(difference))
    difference = F.relu(model.f_diff3(difference))
    difference = model.f_diff4(difference)
    total = F.relu(model.f_sum1(total))
    total = F.relu(model.f_sum2(total))
    total = F.relu(model.f_sum3(total))
    total = model.f_sum4(total)
    prediction = torch.abs(difference * total)[:, 0]  # non-negative

    identical = torch.isclose(first, second, rtol=1e-5, atol=1e-6).all(dim=1)
    return torch.where(identical, torch.zeros_like(prediction), prediction)


# %%
if "get_ipython" in globals() and data_info is not None:
    # Guard against the tutorial drifting away from the trained implementation.
    example_pairs = torch.from_numpy(
        np.array(
            np.load(config.data_dir / "train_features.npy", mmap_mode="r")[:3],
            dtype=np.float32,
        )
    )
    torch.manual_seed(config.seed)
    architecture_models = {
        "PFN": PFN(4, latent_dim=8, phi_hidden_dim=16, f_hidden_dim=16).eval(),
        "MA-PFN": MAPFN(4, latent_dim=8, phi_hidden_dim=16, f_hidden_dim=16).eval(),
    }
    with torch.inference_mode():
        architecture_check = {
            "PFN max |written-out - class|": float(
                torch.max(torch.abs(
                    stock_pfn_forward(architecture_models["PFN"], example_pairs)
                    - architecture_models["PFN"](example_pairs)
                ))
            ),
            "MA-PFN max |written-out - class|": float(
                torch.max(torch.abs(
                    metric_aware_forward(architecture_models["MA-PFN"], example_pairs)
                    - architecture_models["MA-PFN"](example_pairs)
                ))
            ),
        }
    assert max(architecture_check.values()) < 1e-6
    display(architecture_check)


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
if "get_ipython" in globals() and data_info is not None:
    sample_target = torch.from_numpy(
        np.array(
            np.load(config.data_dir / "train_targets.npy", mmap_mode="r")[:4],
            dtype=np.float32,
        )
    )
    sample_prediction = sample_target * torch.tensor([0.8, 1.2, 0.9, 1.1])
    written_out = hybrid_emd_loss_written_out(
        sample_prediction,
        sample_target,
        config.mae_weight,
        config.mae_scale,
    )
    reusable_loss = HybridEMDLoss(
        mae_weight=config.mae_weight,
        mae_scale=config.mae_scale,
        loss="hybrid",
    )
    reusable = reusable_loss.components(sample_prediction, sample_target)
    for expected, actual in zip(written_out, reusable):
        torch.testing.assert_close(expected, actual)
    display({
        "objective_used_for_training": reusable_loss.description,
        "example_hybrid_objective": float(reusable[0]),
        "example_mape": float(reusable[1]),
        "example_mae_gev": float(reusable[2]),
        "max_abs_written_out_vs_class": max(
            float(torch.abs(expected - actual))
            for expected, actual in zip(written_out, reusable)
        ),
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
# path that rebuilds and re-encodes every tagged pair. After training, an
# explicit cache cell below performs these operations and checks its predictions
# against ordinary paired inference.


# %% [markdown]
# ## Train, benchmark, and plot
#
# The workflow trains both models with the displayed hybrid loss, selects their
# best checkpoints by validation hybrid objective, evaluates the held-out test
# pairs, runs the metric-property benchmarks, and writes three plots plus
# machine-readable JSON/NPZ results.

# %%
if "get_ipython" in globals() and data_info is not None:
    results = run_workflow(config)


# %%
if "get_ipython" in globals() and results is not None:
    display({
        "loss": {
            "name": config.loss,
            "mae_weight": config.mae_weight,
            "mae_scale_gev": config.mae_scale,
        },
        "training_at_selected_checkpoint": {
            name: {
                "epoch": history["best_epoch"],
                "validation_objective": history["best_val_objective"],
                "validation_mape": history["best_val_mape"],
                "validation_mae_gev": history["best_val_mae"],
            }
            for name, history in results["training"].items()
        },
        "inference_mode": results["benchmark"]["inference_mode"],
        "data_kind": data_info["kind"],
        "accuracy": results["accuracy"],
        "metric_properties": results["metric_properties"],
    })
    training_plot = config.output_dir / "training_curves.png"
    if training_plot.is_file():
        display(NotebookImage(filename=str(training_plot)))


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
if "get_ipython" in globals() and results is not None:
    device = select_device(config.device)
    held_out_events = reconstruct_split_events(config.data_dir)
    event_tensor = torch.from_numpy(np.ascontiguousarray(held_out_events)).to(device)
    first_indices_np = np.array([0, 0, 1, 2, 3, 5, 8, 13], dtype=np.int64)
    second_indices_np = np.array([1, 2, 3, 4, 5, 8, 13, 21], dtype=np.int64)
    first_indices = torch.from_numpy(first_indices_np).to(device)
    second_indices = torch.from_numpy(second_indices_np).to(device)

    cache_check = {}
    for name, label in (("ma_pfn", "MA-PFN"), ("pfn", "PFN")):
        model = load_checkpoint(config.output_dir / name / "best_model.pt", device)
        cache = cache_events_explicitly(model, event_tensor)
        cached = score_cached_pairs_explicitly(
            model, cache, first_indices, second_indices
        ).cpu().numpy()
        ordinary = predict_event_pairs(
            model,
            held_out_events,
            first_indices_np,
            second_indices_np,
            config.batch_size,
            device,
        )
        max_difference = float(np.max(np.abs(cached - ordinary)))
        assert np.allclose(cached, ordinary, rtol=2e-5, atol=2e-5)
        cache_check[label] = {
            "unique_events": len(held_out_events),
            "latent_shape": list(cache[0].shape),
            "event_encoding_passes": cache[2],
            "checked_pairs": len(cached),
            "max_abs_cached_vs_ordinary_gev": max_difference,
        }
    display(cache_check)


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
if "get_ipython" in globals() and results is not None:
    throughput = benchmark_inference_throughput(
        config,
        events_per_bank=throughput_events_per_bank,
        repetitions=throughput_repetitions,
    )


# %% [markdown]
# The loss plot is shown above. It and the other figures are also saved in
# `results_notebook/` as `training_curves.png`, `accuracy_benchmark.png`, and
# `metric_benchmarks.png`.


# %%
if __name__ == "__main__" and "get_ipython" not in globals():
    run_workflow(parse_args())
