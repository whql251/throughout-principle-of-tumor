"""Inference exports; importing this module does not run the workflow."""

import json
import numpy as np
import pandas as pd
import torch

from .data import ROOT, load_data, output_directory
from .models import load_model


def predict(workflow, output, data=None, checkpoint=None):
    x, names, features = load_data(workflow, data)
    model = load_model(workflow, checkpoint or ROOT / "models" / f"{workflow}_dec.pth")
    with torch.no_grad():
        latent = model.autoencoder.encode(x).numpy()
        probabilities = model(x).numpy()
    labels = probabilities.argmax(axis=1)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1):
        raise ValueError("Invalid DEC probabilities")
    output = output_directory(output)
    frame = pd.DataFrame(latent, columns=[f"latent_{i+1}" for i in range(latent.shape[1])])
    frame.insert(0, "Sample_ID", names)
    frame["Cluster_Assignment"] = labels
    # Biological names are defined only for the two bundled manuscript models.
    if checkpoint is None:
        pattern = {"precancer": {0: "DP1", 1: "DP2", 2: "DP3", 3: "DP4"},
                   "deconvolution": {0: "IP1", 1: "IP3", 2: "IP2"}}[workflow]
        frame["Biological_Pattern"] = [pattern[int(label)] for label in labels]
    frame.to_csv(output / "latent.csv", index=False)
    frame.to_excel(output / "latent.xlsx", index=False)
    scores = pd.DataFrame(probabilities, columns=[f"q_{i}" for i in range(probabilities.shape[1])])
    scores.insert(0, "Sample_ID", names)
    scores.to_csv(output / "probabilities.csv", index=False)
    processed = pd.DataFrame(x.numpy(), columns=features)
    processed.insert(0, "Sample_ID", names)
    processed.to_csv(output / "preprocessed_features.csv", index=False)
    counts = {str(i): int((labels == i).sum()) for i in range(probabilities.shape[1])}
    (output / "summary.json").write_text(json.dumps({
        "workflow": workflow, "samples": len(names), "input_dimensions": x.shape[1],
        "latent_dimensions": latent.shape[1], "cluster_counts": counts,
        "label_convention": "Zero-based checkpoint cluster indices",
    }, indent=2) + "\n")
    return frame
