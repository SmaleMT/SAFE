# SAFE

This repository contains the main implementation of **SAFE: A Screening-Guided
Adaptive Fusion Ensemble for Imbalanced Medical Image Classification**.

The repository focuses on the proposed SAFE method. Code for baseline
comparisons, ablation studies, significance tests, comparative figures, and
additional experiments presented later in the paper is not included.

## Pipeline

SAFE is run in three stages:

1. `train_gan.py` trains the conditional SN-GAN with a projection discriminator
   and DiffAugment. The lowest-FID checkpoint produces the minority-class
   candidate pool.
2. `train_screeners.py` runs CHRE, CHRE-FA, CHRE-MMR, and CHRE-FA-MMR, then
   trains the four branch classifiers and the real-data baseline classifier.
3. `evaluate_safe.py` applies branch-score-guided adaptive entropy fusion and
   reports AUC, ACC, macro F1, and macro Recall.

## Datasets

The datasets are obtained from [MedMNIST v2](https://medmnist.com/) through its
official Python package. The first run downloads the selected dataset to the
local MedMNIST cache (normally `~/.medmnist`), so raw dataset files are not
stored in this repository. The official training, validation, and test splits
are used without redefining the split. Training and screening use the training
and validation data, while the test split is accessed only during final
evaluation.

The supported dataset identifiers are `bloodmnist`, `breastmnist`,
`dermamnist`, `organcmnist`, `organsmnist`, and `pneumoniamnist`. Select a
dataset with the required `--dataset` argument and use the same identifier in
all three stages. Each dataset has an independent output directory under
`artifacts/<dataset>/`.

## Installation

Python 3.9 or later is required.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use the PyTorch installation command recommended for the CUDA version on the
target machine when GPU support is required.

## Usage

Run the three stages separately from the repository root. Wait for each command
to finish before starting the next one.

Step 1: train the GAN and generate the candidate pool.

```bash
python train_gan.py --dataset organcmnist
```

Step 2: screen the generated samples and train the classifiers.

```bash
python train_screeners.py --dataset organcmnist
```

Step 3: evaluate the trained ensemble.

```bash
python evaluate_safe.py --dataset organcmnist
```

Intermediate artifacts are stored under `artifacts/<dataset>/` and are ignored
by Git. Final evaluation writes `artifacts/<dataset>/metrics.json` containing
only:

```json
{
  "ACC": 0.0,
  "AUC": 0.0,
  "F1": 0.0,
  "Recall": 0.0
}
```
