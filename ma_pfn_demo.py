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
# The short notebook is the demo. Model definitions
# live in [`models.py`](models.py), while data loading, training, metrics, and
# plots live in [`utils.py`](utils.py).

# %%
"""Train and benchmark a minimal MA-PFN and stock-PFN comparison."""

from pathlib import Path

from utils import WorkflowConfig, parse_args, run_workflow


# %% [markdown]
# ## What changes in MA-PFN?
#
# Both networks use the same particle-level encoder. The stock PFN sums all
# tagged particle embeddings and applies an unconstrained regression head.
# MA-PFN instead pools the two events separately and builds its prediction from
# exchange-invariant latent sums and absolute differences. A bias-free
# difference branch and an absolute-valued output guarantee non-negativity,
# exchange symmetry, and zero self-distance.

# %% [markdown]
# ## Configure the demonstration
#
# All experimental choices are exposed below so a reader can change them in one
# place. The committed values define a short demonstration. Comments identify
# the release-scale values where they differ. Pair limits select reproducible
# random subsets; use `None` for every available training or validation pair.
# `stage` can be `"all"`, `"train"`, or `"benchmark"`.

# %%
config = WorkflowConfig(
    # Data and workflow
    data_dir=Path("data"),
    output_dir=Path("results_notebook"),
    stage="all",
    device="auto",  # automatically use CUDA when available
    seed=12_345,

    # Architecture (shared wherever possible for a controlled comparison)
    latent_dim=64,
    phi_hidden_dim=100,
    f_hidden_dim=100,

    # Optimization
    epochs=5,  # release scale: 500
    patience=0,  # release scale: 50; 0 disables early stopping
    batch_size=1_024,
    learning_rate=1e-4,
    num_workers=0,
    preload=False,  # True is faster if the selected arrays fit in host RAM

    # Reproducible demo subsets and final benchmarks
    max_train_pairs=50_000,  # release scale: None
    max_val_pairs=10_000,  # release scale: None
    max_test_pairs=20_000,  # release scale: 100_000
    metric_samples=2_000,  # release scale: 20_000
    tolerance=1e-3,  # GeV; pass/fail tolerance for metric properties
    make_plots=True,
)


# %% [markdown]
# ## Train, benchmark, and plot
#
# The workflow trains both models, saves their best checkpoints, evaluates the
# held-out test pairs, runs the metric-property benchmarks, and writes three
# plots plus machine-readable JSON/NPZ results.

# %%
if "get_ipython" in globals():
    results = run_workflow(config)


# %%
if "get_ipython" in globals():
    {
        "accuracy": results["accuracy"],
        "metric_properties": results["metric_properties"],
    }


# %% [markdown]
# The plots are saved in `results_notebook/` as `training_curves.png`,
# `accuracy_benchmark.png`, and `metric_benchmarks.png`.


# %%
if __name__ == "__main__" and "get_ipython" not in globals():
    run_workflow(parse_args())
