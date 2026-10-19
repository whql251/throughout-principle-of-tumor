# MLP phenotype associations

MLP model linking driver mutations and early differentiation patterns to tumor phenotypes.

## Contents

```text
MLP/
  script/           phenotype prediction and score summarization
  data/             phenotype input workbooks
  models/           trained main-model checkpoint
  requirements.txt
```

## Setup

Use Python 3.10. Run commands from the MLP folder. A GPU is not required.

```bash
python -m pip install -r requirements.txt
```

## Usage

Predict phenotype scores and summarize the results:

```bash
python script/generate_results.py
python script/process_results.py
```

By default, `generate_results.py` loads `models/trained_model.pth` without further training. `process_results.py` normalizes and summarizes the predictions. Outputs are saved as `results/predicted_scores.xlsx` and `results/processed_results.xlsx`.

## Training

Train or resume fitting on the full input dataset:

```bash
python script/generate_results.py --train --train_epochs 3200
```

Training may update `models/trained_model.pth`. `--batch_size` defaults to 32.

## Notes

Input and output paths are resolved relative to the MLP folder. Output folders are created automatically.
