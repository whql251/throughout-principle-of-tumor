# DEC clustering

DEC models for precancerous differentiation patterns (Fig. 4B) and tumor immune patterns (Fig. 5H).

## Contents

```text
Clustering/
  scripts/          data loading, models, prediction, visualization, and training
  data/             input workbooks
  models/           trained DEC checkpoints
  requirements.txt
```

## Setup

Use Python 3.10. Run commands from the Clustering folder. A GPU is not required.

```bash
python -m pip install -r requirements.txt
```

## Usage

Predict both datasets using the trained models:

```bash
python -m scripts predict
```

Outputs are saved under `results/precancer/` and `results/deconvolution/`, including latent features, cluster labels, and probabilities. Use `--workflow precancer` or `--workflow deconvolution` to select one dataset, and `--output PATH` to change the output folder.

| Dataset | Samples | Input / latent dimensions | Cluster → pattern |
|---|---:|---:|---|
| precancer | 73 | 17 / 15 | 0→DP1, 1→DP2, 2→DP3, 3→DP4 |
| deconvolution | 52 | 33 / 33 | 0→IP1, 1→IP3, 2→IP2 |

Create UMAP and PCA plots:

```bash
python -m scripts plot --seed 42
```

## Training

Train a new model with example settings:

```bash
python -m scripts train --workflow precancer --output results/new_model --pretrain-epochs 300 --epochs 1501
```

Use a new or empty output folder. Weights are saved as `dec.pth`; use `predict --workflow precancer --checkpoint PATH` to load them. Check the cluster-to-pattern mapping when using newly trained weights.

## Notes

The `deconvolution` workflow clusters immune abundance tables; upstream RNA-seq deconvolution is separate. Stage-two optimization uses KL loss for precancer and KL plus center distance for immune clustering. UMAP projections can vary between runs.

DEC implementation adapted from [Deepayan137/DeepClustering](https://github.com/Deepayan137/DeepClustering).
