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
# and the four metric properties: non-negativity, identity, symmetry, and the
# triangle inequality.
#
# The intentionally short notebook is the experiment recipe. Model definitions
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
# exchange symmetry, and zero self-distance. Triangle inequality is measured,
# not imposed.

# %% [markdown]
# ## Configure the demonstration
#
# These limits make the notebook quick to inspect and test. Set the pair limits
# to `None`, increase `epochs`, and raise `metric_samples` for the full release
# run. The command-line interface exposes the same settings via
# `python ma_pfn_demo.py --help`.

# %%
if "get_ipython" in globals():
    config = WorkflowConfig(
        data_dir=Path("data"),
        output_dir=Path("results_notebook"),
        epochs=5,
        patience=0,
        max_train_pairs=50_000,
        max_val_pairs=10_000,
        max_test_pairs=20_000,
        metric_samples=2_000,
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
