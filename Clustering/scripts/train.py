"""Train new CPU models without changing the released checkpoints."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader

from .data import ROOT, load_data, output_directory
from .models import CONFIG, DEC


def _output_directory(path):
    output = Path(path).resolve()
    root = ROOT.resolve()
    if output == root or output in root.parents:
        raise ValueError("Choose a separate training output directory")
    output = output_directory(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be a new or empty directory")
    return output


def centreness_loss(embeddings, assignments, centers):
    """Source deconvolution term: sample-mean squared distance to a center."""
    selected = centers[assignments.argmax(dim=1)]
    return torch.sum((embeddings - selected) ** 2) / embeddings.shape[0]


def pretrain(data, autoencoder, epochs, log):
    """Shared source denoising autoencoder: Gaussian noise with std=0.2."""
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(data, batch_size=128, shuffle=True)
    best = float("inf")
    for epoch in range(epochs):
        total, count = 0.0, 0
        for batch in loader:
            noisy = batch + torch.randn(batch.size()) * 0.2
            loss = nn.functional.mse_loss(autoencoder(noisy), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite pretraining loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        value = total / count
        best = min(best, value)
        log.append({"stage": "pretrain", "epoch": epoch, "total_loss": value,
                    "reconstruction_loss": value, "kl_loss": None, "centreness_loss": None})
    return best


def train_workflow(workflow, output, pretrain_epochs, epochs, seed):
    """Save a newly trained model, predictions, loss log, and settings."""
    if workflow not in CONFIG:
        raise ValueError(f"Unknown workflow: {workflow}")
    if pretrain_epochs < 1 or epochs < 1:
        raise ValueError("Epoch counts must be positive")
    x, names, _ = load_data(workflow)
    output = _output_directory(output)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = DEC(*CONFIG[workflow])
        log = []
        best_pretrain = pretrain(x, model.autoencoder, pretrain_epochs, log)
        torch.save({"state_dict": model.autoencoder.state_dict(), "best": best_pretrain,
                    "epoch": pretrain_epochs - 1}, output / "autoencoder.pth")
        with torch.no_grad():
            features = model.autoencoder.encode(x).numpy()
        centers = KMeans(n_clusters=CONFIG[workflow][2], random_state=0, n_init=10).fit(features).cluster_centers_
        model.clusteringlayer.cluster_centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        learning_rate = 0.01 if workflow == "deconvolution" else 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        best = float("inf")
        for epoch in range(epochs):
            model.train()
            q = model(x)
            target = model.target_distribution(q).detach()
            kl = nn.functional.kl_div(q.log(), target, reduction="sum")
            if workflow == "deconvolution":
                center_loss = centreness_loss(model.autoencoder.encode(x), q,
                                              model.clusteringlayer.cluster_centers)
                loss = kl + 2 * center_loss
            else:
                kl = kl / len(x)
                center_loss = torch.zeros((), dtype=x.dtype)
                loss = kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite clustering loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            best = min(best, value)
            log.append({"stage": "clustering", "epoch": epoch, "total_loss": value,
                        "reconstruction_loss": None, "kl_loss": float(kl.detach()),
                        "centreness_loss": float(center_loss.detach())})
        checkpoint = output / "dec.pth"
        torch.save({"state_dict": model.state_dict(), "best": best, "epoch": epochs - 1}, checkpoint)
        pd.DataFrame(log).to_csv(output / "training_log.csv", index=False)
        # Recompute exports after the final update so they match the saved model.
        from .pipeline import predict
        frame = predict(workflow, output / "prediction", checkpoint=checkpoint)
        summary = {
            "workflow": workflow, "seed": seed,
            "samples": len(names), "input_dimensions": CONFIG[workflow][0],
            "latent_dimensions": CONFIG[workflow][1], "clusters": CONFIG[workflow][2],
            "pretrain_epochs": pretrain_epochs, "clustering_epochs": epochs,
            "pretraining": {"optimizer": "Adam", "learning_rate": 1e-3,
                            "weight_decay": 1e-5, "batch_size": 128, "noise_std": 0.2,
                            "loss": "mean squared reconstruction error"},
            "clustering": {"optimizer": "SGD", "learning_rate": learning_rate,
                           "momentum": 0.9, "kmeans_random_state": 0, "kmeans_n_init": 10,
                           "loss": ("KL sum + 2 * sample-mean squared center distance"
                                    if workflow == "deconvolution" else "KL sum / sample count")},
            "cluster_counts": {str(k): int((frame.Cluster_Assignment == k).sum())
                               for k in range(CONFIG[workflow][2])},
            "checkpoint_selection": "final iteration", "historical_equality_claimed": False,
            "label_convention": "New run indices; biological DP/IP names require independent interpretation",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    print(f"{workflow}: trained a new model and saved {checkpoint}")
    return summary
