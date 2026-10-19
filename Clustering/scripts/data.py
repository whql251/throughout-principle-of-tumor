"""Original workbook parsing and cohort-level preprocessing."""

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import torch


GENES = ["Trp53", "Nf1", "Rb1", "Pten", "Pik3cg", "Cdkn2a/2b", "EgfrvIII"]
MUTATIONS = {
    "P78": [0,0,0,0,0,0,0], "P74": [1,1,0,0,0,0,0],
    "P81": [1,0,0,1,0,0,0], "P112": [1,0,1,0,0,0,0],
    "P113": [0,0,1,1,0,0,0], "P114": [0,1,0,1,0,0,0],
    "P115": [0,0,0,0,0,1,0], "P116": [0,0,0,1,0,1,0],
    "P117": [0,1,0,0,0,1,0], "P118": [1,0,1,1,0,0,0],
    "P119": [1,1,1,0,0,0,0], "P120": [0,1,1,1,0,0,0],
    "P121": [1,1,0,1,0,0,0], "P122": [0,0,1,1,0,1,0],
    "P123": [1,0,0,1,0,1,0], "P124": [0,1,0,1,0,1,0],
    "P125": [0,1,1,0,0,0,0], "P126": [0,0,1,1,1,0,0],
    "P132": [1,0,0,1,1,0,0], "P135": [0,1,0,1,1,0,0],
    "P136": [1,1,0,0,1,0,0], "P140": [0,1,0,1,0,0,1],
    "P201": [1,1,1,1,0,0,0],
}
ROOT = Path(__file__).resolve().parents[1]


def output_directory(path):
    """Keep generated results away from the released source assets."""
    path = Path(path).resolve()
    if path == ROOT or any(path == ROOT / name or ROOT / name in path.parents
                           for name in ("data", "models", "scripts")):
        raise ValueError("Choose a results directory outside scripts/data/models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_precancer(path):
    workbook = openpyxl.load_workbook(path)
    rows, names = [], []
    genotype = animal = None
    for row in list(workbook.active.values)[2:]:
        if row[0] is not None:
            genotype = str(row[0]).split(" ")[0]
        if row[2] is not None:
            animal = str(row[2])
        values = []
        if row[3] is not None:
            pattern = str(row[3])
            if len(pattern) != 7 or set(pattern) - {"0", "1"}:
                raise ValueError(f"Invalid mutation pattern: {pattern!r}")
            values.extend(float(v) for v in pattern)
        values.extend(float(v) for v in row[4:] if v is not None)
        if values:
            if len(values) != 17 or genotype is None or animal is None:
                raise ValueError(f"Expected 17 features and sample metadata: {genotype}, {animal}")
            rows.append(values)
            names.append(f"{genotype}_{animal}")
    workbook.close()
    x = torch.tensor(rows, dtype=torch.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Precancer workbook must contain at least two samples")
    # torch.std uses the sample standard deviation (ddof=1), as in load_mnist.
    mean, std = x.mean(dim=0), x.std(dim=0)
    if torch.any(std == 0):
        raise ValueError("Zero-variance column: original z-score preprocessing is undefined")
    x = (x - mean) / std
    feature_names = GENES + [f"Differentiation_Feature_{i+1}" for i in range(10)]
    return x, names, feature_names


def load_deconvolution(path):
    table = pd.read_excel(path)
    table = table.iloc[1:]  # Second header/metadata row, as in load_excel_data.
    table = table.loc[table.iloc[:, 0].map(lambda v: str(v).split("_")[0] != "P200")]
    names = table.iloc[:, 0].tolist()
    if not names or any(not isinstance(v, str) for v in names):
        raise ValueError("Missing or invalid Sample_ID")
    numeric = table.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0)
    if numeric.shape[1] < 26:
        raise ValueError("Expected at least 26 immune abundance columns")
    mutations = [MUTATIONS.get(v.split("_")[0], [0] * 7) for v in names]
    x = torch.tensor(np.hstack([numeric.values[:, :26], mutations]), dtype=torch.float32)
    low, high = x[:, :26].min(dim=0)[0], x[:, :26].max(dim=0)[0]
    span = high - low
    span = torch.where(span == 0, torch.ones_like(span), span)
    x = torch.cat([(x[:, :26] - low) / span, x[:, 26:]], dim=1)
    feature_names = ([f"Relative_Feature_{i+1}" for i in range(13)]
                     + [f"Absolute_Feature_{i+1}" for i in range(13)]
                     + [f"Mutation_Feature_{i+1}" for i in range(7)])
    return x, names, feature_names


def load_data(workflow, path=None):
    path = Path(path) if path else ROOT / "data" / f"{workflow}.xlsx"
    loader = load_precancer if workflow == "precancer" else load_deconvolution
    x, names, features = loader(path)
    if not torch.isfinite(x).all():
        raise ValueError("Preprocessed features contain NaN or infinity")
    return x, names, features
