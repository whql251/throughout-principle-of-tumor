# Tumor phenotype analysis

DEC clustering of precancerous differentiation and tumor immune patterns, and MLP modeling of their associations with driver mutations and tumor phenotypes.

## Contents

```text
final_version/
  Clustering/       DEC prediction, visualization, and training
  MLP/              phenotype prediction, score summarization, and model fitting
```

## Setup

Use Python 3.10. Each module provides its own `requirements.txt`. Follow the setup instructions in the corresponding README.

## Usage

- [Clustering](Clustering/README.md): identify differentiation and immune patterns using the trained DEC models.
- [MLP](MLP/README.md): predict and summarize phenotype associations using the trained main model.
