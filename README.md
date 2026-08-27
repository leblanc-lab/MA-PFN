# Minimal MA-PFN example

This directory is a self-contained example of the
metric-aware particle flow network (MA-PFN) used to regress energy mover's
distance (EMD). It trains MA-PFN and the original unconstrained PFN with the
same data split and optimizer settings, then compares their held-out accuracy
and metric properties. Both models use the constructed dimensionless objective
`MAPE + 0.25 * MAE / 90 GeV`, matching the production hybrid-loss controls.

This example implements the selected symmetric joint-head MA-PFN. Its shared
encoder maps `4 -> 100 -> 100 -> 64`. A `128 -> 100 -> 100 -> 100 -> 1` head
receives the pooled latent sum and signed difference, is evaluated at both
event orientations, and has its two outputs averaged. The prediction is the
mean absolute latent separation multiplied by the softplus of that symmetric
head output. This gives non-negativity, zero self-distance, and exchange
symmetry by construction. At these dimensions, MA-PFN has **50,265** trainable
parameters and the matched stock PFN has **43,865**.

## Contents

- `ma_pfn_demo.py`: readable Jupytext source for the notebook.
- `ma_pfn_demo.ipynb`: generated tutorial notebook with a short default run.
- `run_demo.py`: command-line entry point.
- `ma_pfn_tutorial.npz`: checksum-verified 40 MB subset of the real
  released event pairs and exact EMD targets.
- `models.py`: side-by-side MA-PFN and stock-PFN model definitions.
- `utils.py`: subset extraction, data loading, training, evaluation, plotting,
  and CLI helpers.
- `test_demo.py`: fast checks for data, loss, parameter counts, structural
  properties, and the cached model path.
- `make_notebook.py`: local Jupytext wrapper used by CI.
- `jupytext.toml`: declares the paired `ipynb,py:percent` formats.
- `.github/workflows/sync-notebook.yml`: regenerates and commits the notebook
  after the Python source is pushed.
- `requirements.txt`: minimal runtime and notebook dependencies.
- `zenodo/make_tutorial_subset.py`: deterministic builder for the compact
  archive from the six full release arrays.
- `zenodo/DATASET_DESCRIPTION.md`: editable metadata draft for the data record.

## Data layout

No data download is required to run the notebook tutorial. On **Run All**, the
notebook uses the six arrays in `data/` when they are present. When none is
present, it extracts the bundled `ma_pfn_tutorial.npz` into
`results_notebook/tutorial_data/` after verifying its SHA-256 checksum.

To use the released sample, place its six NumPy arrays in `data/` (or pass
another directory with `--data-dir` to the command-line workflow):

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

If only some of the six files exist, the notebook stops before training and
lists the missing paths. This avoids silently mixing a partial release with the
tutorial subset.

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

```bash
python run_demo.py \
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

For the production training configuration, omit the training and validation
pair limits. The defaults are 700 epochs, patience 50, batch size 1,024, AdamW
with learning rate `1e-4` and zero weight decay, and seed 23,411:

```bash
python run_demo.py \
  --data-dir data \
  --output-dir results \
  --device cuda \
  --preload
```

The two models train sequentially, making the command work on a one-GPU
machine. Progress is printed once per epoch. The validation set controls early
stopping using the hybrid objective; the test set is first touched by the final
benchmark. The objective, MAPE, and MAE in GeV are all logged separately. The
loss can be varied explicitly with `--loss`, `--mae-weight`, and `--mae-scale`;
optimizer weight decay can be varied with `--weight-decay`.

The command-line defaults of 100,000 held-out pairs and 20,000 metric samples
are deliberately smaller evaluation workloads for this minimal example. They
must not be reported as reproducing the paper's all-798,216-pair response study
or its one-million-triplet metric-property study. To request those paper-scale
sample counts from checkpoints trained by this workflow, pass
`--max-test-pairs 0 --metric-samples 1000000`.

Training and benchmarking can be separated without retraining:

```bash
python run_demo.py --stage train --data-dir data --output-dir results --device cuda
python run_demo.py --stage benchmark --data-dir data --output-dir results --device cuda
```

Architecture dimensions and the explicit `joint` or `baseline` architecture
identifier are stored in each checkpoint, so a benchmark-only command
reconstructs the trained models without repeating those arguments.

Held-out inference defaults to the cached-latent path used by the timing
benchmarks. Each unique test event is transferred and encoded before pair
evaluation; pair batches then gather resident latents and run only the
regression head. MA-PFN uses one shared latent bank. Because the stock PFN
learns the event-identity tag, it uses separate first-role and second-role
latent banks. Pass `--no-cache-inference` to reproduce the legacy path that
rebuilds and re-encodes every tagged pair tensor.

The notebook also contains a small throughput cell comparing resident pair
encoding with cached inference for identical cross-event pairs. It keeps the
event tensors and indices on the selected device for both paths, matching the
timing-study contract. The cell reports cold cached throughput (including
setup), resident cached throughput, and both speedups for each model in a
two-panel figure, then writes the complete timings to
`inference_throughput.json`. This is an
illustrative sanity check; use the paper's controlled timing suite for reported
performance numbers.

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
├── inference_throughput.png  # when the notebook timing cell is run
├── benchmark_arrays.npz
├── inference_throughput.json  # when the notebook timing cell is run
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

Open `ma_pfn_demo.ipynb`, edit the `config` cell, and run
all cells. A clean checkout automatically uses the real-data subset described
above. Its committed defaults are intentionally small, with release-scale
values noted next to the settings that differ.

The notebook includes executable, written-out versions of the stock-PFN and
MA-PFN forward passes, writes out the constructed hybrid-loss equation and
implementation, and displays the objective/MAPE/MAE learning curves inline.
It also explicitly encodes unique events, constructs the one-bank MA-PFN and
two-role PFN caches, gathers pair latents, and times cached inference. Training
and plotting details remain in helpers so they do not obscure those ideas.

`ma_pfn_demo.py` is the source of truth for the generated notebook. To
regenerate the notebook locally after editing its source:

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

Run the tutorial regression tests with:

```bash
python -m unittest -v test_demo.py
```

To reproduce the compact archive from the full local arrays:

```bash
python zenodo/make_tutorial_subset.py \
  --source-dir /path/to/full/release/arrays
```

The builder is deterministic: the default invocation must produce SHA-256
`84a0f9c2bb0d1ff793a2ffc2c2e1ee4671d46fe1d61a07f83d93bd0f4bd1b1fb`.
