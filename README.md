# Minimal MA-PFN example

This directory is a self-contained, release-oriented example of the
metric-aware particle flow network (MA-PFN) used to regress energy mover's
distance (EMD). It trains MA-PFN and the original unconstrained PFN with the
same data split and optimizer settings, then compares their held-out accuracy
and metric properties.

The notebook is deliberately short: it contains only the experiment narrative,
configuration, workflow call, and result summary. The model definitions and
reusable experiment machinery live in ordinary Python modules, so the
command-line and interactive versions still share one implementation.

## Contents

- `ma_pfn_demo.py`: concise notebook source and command-line entry point.
- `ma_pfn_demo.ipynb`: generated minimal notebook with a short default run.
- `models.py`: side-by-side MA-PFN and stock-PFN model definitions.
- `utils.py`: data loading, training, evaluation, plotting, and CLI helpers.
- `make_notebook.py`: local Jupytext wrapper used by CI.
- `jupytext.toml`: declares the paired `ipynb,py:percent` formats.
- `.github/workflows/sync-notebook.yml`: regenerates and commits the notebook
  after the Python source is pushed.
- `requirements.txt`: minimal runtime and notebook dependencies.
- `zenodo/DATASET_DESCRIPTION.md`: editable metadata draft for the data record.

## Data layout

Place the six released NumPy arrays in `data/` (or pass another directory with
`--data-dir`):

```text
data/
├── train_features.npy
├── train_targets.npy
├── val_features.npy
├── val_targets.npy
├── test_features.npy
└── test_targets.npy
```

For every split:

- features have shape `(number of pairs, padded particles, 4)` and contain
  `(pT [GeV], eta, phi [rad], event_id)`;
- `event_id` is `-1` for the first event and `+1` for the second;
- zero kinematics indicate padding (the padding row still carries its event
  ID); and
- targets have shape `(number of pairs,)` and contain exact EMD in GeV.

The supplied splits are event-disjoint: no event used to form a training pair
appears in validation or test. Within each split, rows contain all unordered
pairs in standard combinations order. That ordering lets the example recover
the 1,264 held-out test events from `test_features.npy` for the metric tests.

The current release arrays have the following inventory:

| Split | Source events | Event pairs | Feature shape | Target dtype |
| --- | ---: | ---: | --- | --- |
| Train | 3,577 | 6,395,676 | `(6,395,676, 176, 4)` | `float32` |
| Validation | 1,264 | 798,216 | `(798,216, 174, 4)` | `float32` |
| Test | 1,264 | 798,216 | `(798,216, 152, 4)` | `float32` |

Feature arrays in the current data preparation are `float64`; the loader
converts them to `float32`, matching training. Files are memory mapped by
default, so the full data set does not need to fit in RAM. On a machine with
roughly 24 GB of available host memory, add `--preload` to copy the selected
training and validation arrays to float32 RAM once; this substantially improves
random-shuffle throughput on network filesystems.

## Setup

Python 3.10 or newer is recommended. Create an isolated environment and
install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for the local CUDA
version by following the PyTorch installation selector before installing the
remaining requirements.

## Run it

A short end-to-end smoke run is:

```bash
python ma_pfn_demo.py \
  --data-dir data \
  --output-dir results \
  --device auto \
  --epochs 2 \
  --patience 0 \
  --max-train-pairs 20000 \
  --max-val-pairs 5000 \
  --max-test-pairs 10000 \
  --metric-samples 1000
```

For the release-scale configuration, omit the training and validation pair
limits and use the defaults of 500 epochs, patience 50, batch size 1,024, Adam
learning rate `1e-4`, and seed 12,345:

```bash
python ma_pfn_demo.py \
  --data-dir data \
  --output-dir results \
  --device cuda \
  --preload
```

The two models train sequentially, making the command work on a one-GPU
machine. Progress is printed once per epoch. The validation set controls early
stopping; the test set is first touched by the final benchmark.

Training and benchmarking can be separated without retraining:

```bash
python ma_pfn_demo.py --stage train --data-dir data --output-dir results --device cuda
python ma_pfn_demo.py --stage benchmark --data-dir data --output-dir results --device cuda
```

Architecture dimensions are stored in each checkpoint, so a benchmark-only
command reconstructs the trained models without repeating those arguments.

## Outputs

```text
results/
├── ma_pfn/best_model.pt
├── ma_pfn/history.json
├── pfn/best_model.pt
├── pfn/history.json
├── training_curves.png
├── accuracy_benchmark.png
├── metric_benchmarks.png
├── benchmark_arrays.npz
└── summary.json
```

`metric_benchmarks.png` compares:

1. non-negativity, `d(A,B) >= 0`;
2. identity, `d(A,A) = 0`;
3. symmetry, `d(A,B) = d(B,A)`; and
4. triangle inequality, evaluated as the largest predicted side minus the sum
   of the other two sides for triplets of distinct held-out events.

MA-PFN enforces the first three properties structurally. Triangle inequality
is not built into the architecture and is an empirical benchmark for both
models. `summary.json` records all pass fractions, regression metrics, run
configuration, software versions, and sample counts. The default absolute
tolerance for pass/fail decisions is `1e-3` GeV.

## Notebook

Open `ma_pfn_demo.ipynb`, edit the single, fully explicit `config` cell, and run
all cells. Its committed defaults are intentionally small, with release-scale
values noted next to the settings that differ. `ma_pfn_demo.py` is the source
of truth for the generated notebook; `models.py` and `utils.py` hold the
imported implementation. To regenerate the notebook locally after editing its
source:

```bash
python -m pip install jupytext==1.19.5
python make_notebook.py
```

The GitHub Actions workflow runs the same command whenever `ma_pfn_demo.py`,
`models.py`, `utils.py`, the wrapper, or the Jupytext configuration is pushed.
If the generated notebook changed, the workflow commits it back to the pushed
branch as `github-actions[bot]`. The workflow requests only `contents: write`;
the repository must allow GitHub Actions to write to the target branch.
Protected branches that require pull requests will reject this automatic
commit.

The `.github/workflows` path assumes that the contents of this directory become
the root of the standalone release repository. GitHub will not discover this
workflow while `ma-pfn-minimal/` remains nested inside a different repository.

Do not edit generated notebook code directly. Change the notebook-facing cells
in `ma_pfn_demo.py`; change models or reusable workflow code in `models.py` or
`utils.py`.

## Reproducibility and release checklist

- Upload the dataset directly from OSCAR with
  `./zenodo/upload_to_zenodo.sh`. The script defaults to draft record
  `22099234`, prompts securely for a Zenodo token, verifies file sizes, and
  leaves publication as a manual step. Run it first with `--dry-run` to inspect
  the files without contacting Zenodo. Use `--help` to override the draft ID or
  data directory.
- Upload `zenodo/checksums.sha256` alongside the six arrays.
- Fill every remaining `[TODO]` field in `zenodo/DATASET_DESCRIPTION.md`,
  especially the authors, license, paper DOI, array-derivation details, and
  split provenance.
- Add the chosen code license as `LICENSE`. The data license belongs in the
  Zenodo record and may differ from the code license.
- Run the smoke command in a fresh environment and archive `summary.json` with
  the paper's reference outputs.
- GPU reductions can vary slightly across hardware and PyTorch/CUDA versions;
  the saved seed and environment metadata make those differences auditable.
