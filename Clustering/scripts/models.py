"""Checkpoint-compatible autoencoder and Student-t DEC layers."""

import inspect
from pathlib import Path

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(True))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.ReLU(True))
        # Retain the original aliases: released state dictionaries contain both.
        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.model(x)


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, hidden_dim, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, x):
        distance = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = (1.0 / (1.0 + distance / self.alpha)) ** ((self.alpha + 1) / 2)
        return (numerator.t() / torch.sum(numerator, 1)).t()


class DEC(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_clusters):
        super().__init__()
        self.autoencoder = AutoEncoder(input_dim, hidden_dim)
        self.clusteringlayer = ClusteringLayer(n_clusters, hidden_dim)

    def forward(self, x):
        return self.clusteringlayer(self.autoencoder.encode(x))

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


CONFIG = {"precancer": (17, 15, 4), "deconvolution": (33, 33, 3)}


def load_model(workflow, checkpoint):
    model = DEC(*CONFIG[workflow])
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    saved = torch.load(Path(checkpoint), **kwargs)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model
