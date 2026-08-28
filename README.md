# Minimal MA-PFN example

This repository is a compact, runnable implementation of the Metric-Aware
Particle Flow Network (MA-PFN) used to regress the Energy Mover's Distance
(EMD). It trains MA-PFN alongside a matched, unconstrained Particle Flow
Network (PFN), then compares their held-out accuracy and metric properties.

## Model

Both models operate on padded event pairs with particle features
`(pT, eta, phi, event_id)`. For MA-PFN, the `-1`/`+1` event tag selects one of
two pools but is set to zero before the shared particle encoder:

- encoder: `4 -> 100 -> 100 -> 64`, with two hidden ReLU activations;
- pair features: latent sum `Sigma` and signed difference `Delta`;
- joint head: `128 -> 100 -> 100 -> 100 -> 1`, with hidden ReLUs;
- symmetric scale: the joint head is evaluated at both `Delta` and `-Delta`
  and the outputs are averaged;
- prediction: `mean(abs(Delta)) * softplus(symmetric_scale)`.

Numerically coincident latent pairs are set explicitly to zero. The resulting
model enforces non-negativity, zero self-distance, and exchange symmetry.
Triangle inequality is evaluated empirically rather than imposed by the
architecture.

At these dimensions, MA-PFN has **50,265** trainable parameters and the matched
PFN has **43,865**.

## Data availability

The full event-disjoint training, validation, and test arrays are available to
reviewers through the anonymous data-access link accompanying the submission.
They will be released publicly after peer review.

No download is needed for the tutorial: `ma_pfn_tutorial.npz` contains a
bundled subset of the real event pairs and exact EMD targets. The notebook
extracts it automatically when full arrays are not present.

For the full workflow, place the six arrays in `data/`:

```text
data/
├── train_features.npy
├── train_targets.npy
├── val_features.npy
├── val_targets.npy
├── test_features.npy
└── test_targets.npy
```

Feature arrays have shape `(pairs, padded_particles, 4)` and contain
`(pT [GeV], eta, phi [rad], event_id)`. Zero kinematics mark padding. Targets
have shape `(pairs,)` and contain exact EMD values in GeV. The splits are
event-disjoint, so no source event appears in more than one split.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your CUDA version.

Open `ma_pfn_demo.ipynb` and run all cells for the shortest introduction. Its
defaults are intentionally smaller than the paper configuration and use the
bundled data subset while the full dataset is unavailable.

The command-line workflow is:

```bash
python run_demo.py \
  --data-dir data \
  --output-dir results \
  --device auto
```

Its training defaults match the paper configuration: 700 epochs, patience 50,
batch size 1,024, AdamW with learning rate `1e-4` and zero weight decay, seed
23,411, and the objective

```text
MAPE + 0.25 * MAE / 90 GeV.
```

For a short command-line run, limit the workloads explicitly:

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

Training and benchmarking can also be run separately:

```bash
python run_demo.py --stage train --data-dir data --output-dir results --device cuda
python run_demo.py --stage benchmark --data-dir data --output-dir results --device cuda
```

The default evaluation workloads are smaller than those in the paper. To use
all 798,216 held-out pairs and one million metric-property samples, pass
`--max-test-pairs 0 --metric-samples 1000000`. The tutorial timing cell is only
an illustrative sanity check; use the paper's controlled timing study for
publication-facing performance results.

## Outputs

The workflow writes model checkpoints and histories, training and benchmark
plots, benchmark arrays, and a `summary.json` containing the configuration,
software versions, sample counts, regression metrics, and metric-property pass
fractions.

The metric benchmark covers:

1. non-negativity;
2. zero self-distance;
3. exchange symmetry; and
4. triangle inequality on triplets of distinct held-out events.

Held-out inference caches each unique event embedding and evaluates requested
pairs from those latents. Use `--no-cache-inference` to run the uncached paired
tensor path instead.

## Contributing

When editing the tutorial, change `ma_pfn_demo.py` and regenerate the notebook:

```bash
python make_notebook.py
```
