"""Seeded CPU projections of the released DEC latent representations."""

import hashlib
import json
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from .data import ROOT, load_data, output_directory
from .models import load_model


PAPER_LABELS = {
    "precancer": {0: "DP1", 1: "DP2", 2: "DP3", 3: "DP4"},
    "deconvolution": {0: "IP1", 1: "IP3", 2: "IP2"},
}
COLORS = {
    "DP1": "#D94848", "DP2": "#239EB8", "DP3": "#38A178", "DP4": "#A15EA5",
    "IP1": "#808890", "IP2": "#D94867", "IP3": "#3E78A8",
}


def _save_projection(coordinates, labels, method, workflow, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        for label in sorted(set(labels)):
            mask = labels == label
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=38,
                       color=COLORS[label], alpha=0.9, edgecolors="white",
                       linewidths=0.4, label=f"{label} (n={int(mask.sum())})")
        title = "Precancer differentiation" if workflow == "precancer" else "Tumor immune composition"
        ax.set(title=f"{title}: {method}", xlabel=f"{method} 1", ylabel=f"{method} 2")
        ax.legend(title="Checkpoint groups", frameon=False, loc="best")
        ax.margins(0.12)
        ax.tick_params(length=3, color="#8A939B")
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{method.lower()}.{suffix}", dpi=180)
        plt.close(fig)


def plot_workflow(workflow, output, seed=42):
    """Export fresh UMAP/PCA plots, coordinates and all projection parameters.

    Biological label mapping applies only to the packaged historical weights.
    Seeded projection coordinates are not historical manuscript coordinates.
    """
    from sklearn.decomposition import PCA
    from umap import UMAP

    output = output_directory(output)
    x, names, _ = load_data(workflow)
    checkpoint = ROOT / "models" / f"{workflow}_dec.pth"
    model = load_model(workflow, checkpoint)
    with torch.no_grad():
        latent = model.autoencoder.encode(x).cpu().numpy()
        raw_labels = model(x).argmax(dim=1).cpu().numpy()
    label_mapping = PAPER_LABELS[workflow]
    labels = np.array([label_mapping[int(i)] for i in raw_labels])
    umap_parameters = {
        "n_components": 2, "n_neighbors": min(15, len(names) - 1),
        "min_dist": 0.1, "metric": "euclidean", "init": "spectral",
        "random_state": seed, "transform_seed": seed, "n_jobs": 1,
    }
    umap_coordinates = UMAP(**umap_parameters).fit_transform(latent)
    pca_parameters = {"n_components": 2, "svd_solver": "full"}
    pca = PCA(**pca_parameters)
    pca_coordinates = pca.fit_transform(latent)
    if not (np.isfinite(umap_coordinates).all() and np.isfinite(pca_coordinates).all()):
        raise ValueError("Projection produced non-finite coordinates")
    frame = pd.DataFrame({
        "Sample_ID": names, "Cluster_Assignment": raw_labels, "Paper_Pattern": labels,
        "UMAP_1": umap_coordinates[:, 0], "UMAP_2": umap_coordinates[:, 1],
        "PCA_1": pca_coordinates[:, 0], "PCA_2": pca_coordinates[:, 1],
    })
    frame.to_csv(output / "embeddings.csv", index=False)
    parameters = {
        "workflow": workflow, "seed": seed, "device": "cpu",
        "samples": len(names), "latent_dimensions": int(latent.shape[1]),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "paper_label_mapping": label_mapping, "umap": umap_parameters,
        "pca": pca_parameters,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "versions": {name: version(name) for name in
                     ("umap-learn", "numba", "numpy", "scikit-learn", "matplotlib", "torch")},
        "interpretation": "New projections of archived checkpoint embeddings; no historical coordinate equality claim.",
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    _save_projection(umap_coordinates, labels, "UMAP", workflow, output)
    _save_projection(pca_coordinates, labels, "PCA", workflow, output)
    return frame
