# Integrative MLP-based Analysis of Driver Mutation-to-Tumor Phenotype Associations

This repository contains the code, input tables, trained model, and output tables for the integrative multilayer perceptron (MLP)-based framework used to associate driver mutation combinations and early cellular differentiation patterns with later tumor phenotypes.

The workflow follows the manuscript analysis in three steps: data preprocessing and augmentation, MLP-based model construction and parameter optimization, and data imputation with feature aggregation.

## Contents

- `code/generate_results.py`: trains or resumes the MLP model from `models/trained_model.pth`, then imputes tumor phenotype scores for mutation and differentiation-pattern combinations.
- `code/process_results.py`: aggregates and ranks the imputed phenotype scores into the processed association table.
- `code/run_cross_validation.py`: trains or resumes fold-level MLP models and generates cross-validation prediction outputs.
- `code/process_cross_validation_results.py`: aggregates the fold-level prediction tables and selected feature-pair outputs.
- `data/`: tumor-stage phenotype input tables used as MLP labels, including molecular phenotypes, microenvironmental composition, and survival-related readouts.
- `models/`: MLP checkpoint used by `generate_results.py` for resumed training and score imputation.
- `results/`: imputed and aggregated mutation-to-phenotype association tables.
- `cross_validation/seed_42/`: saved cross-validation outputs.

## Input Files

The scripts expect these files to be present at the repository root:

- `data/molecular_phenotypes.xlsx`: tumor molecular cluster and molecular subtype labels.
- `data/microenvironment_phenotypes.xlsx`: tumor microenvironment proportions and T cell composition readouts.
- `data/survival_phenotypes.xlsx`: tumor incidence and median survival time readouts.
- `models/trained_model.pth`: checkpoint used to resume model fitting before score imputation.

## Output Files

The included output tables are:

- `results/predicted_scores.xlsx`: imputed phenotype scores generated after MLP model fitting.
- `results/processed_results.xlsx`: aggregated association scores for downstream interpretation.

Cross-validation outputs are stored in `cross_validation/seed_42/`, including fold-level prediction tables, trained fold models, selected feature-pair lists, processed tables, and summary figures.

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run commands from the repository root.

Run the full MLP-based association pipeline:

```bash
python code/generate_results.py
python code/process_results.py
```

This first fits the MLP model and imputes phenotype scores for mutation and differentiation-pattern combinations, then aggregates them into `results/processed_results.xlsx`.

Run the cross-validation pipeline:

```bash
python code/run_cross_validation.py
python code/process_cross_validation_results.py
```

This first fits fold-level MLP models and generates cross-validation prediction outputs, then processes the fold-level tables under `cross_validation/seed_42/`.

To run only one step, use the corresponding script directly:

```bash
python code/generate_results.py
python code/process_results.py
python code/run_cross_validation.py
python code/process_cross_validation_results.py
```

The scripts use relative paths, so they should be run from the repository root rather than from inside the `code/` directory.
