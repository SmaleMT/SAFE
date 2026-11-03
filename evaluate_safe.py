from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from train_gan import (
    SUPPORTED_DATASETS,
    MedMNISTDataset,
    classifier_transforms,
    get_dataset_spec,
    make_loader,
    dataset_artifact_dir,
    get_device,
    load_json,
    load_torch,
    save_json,
)
from train_screeners import BRANCHES, load_classifier, predict_probabilities


def standardize_branch_scores(branch_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(branch_scores, dtype=np.float64)
    if scores.shape != (4,):
        raise ValueError("SAFE expects four screening-branch scores")
    standardized = (scores - scores.mean()) / max(scores.std(), 1e-6)
    return np.concatenate([standardized, np.zeros(1, dtype=np.float64)])


def adaptive_entropy_fusion(
    model_probabilities: np.ndarray,
    standardized_branch_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(model_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] != 5:
        raise ValueError("Expected probabilities with shape (5, samples, classes)")
    quality = np.asarray(standardized_branch_scores, dtype=np.float64)
    if quality.shape != (5,):
        raise ValueError("Expected four branch scores followed by the baseline score")

    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    num_classes = probabilities.shape[2]
    entropy = -np.sum(probabilities * np.log(probabilities), axis=2)
    entropy /= np.log(max(num_classes, 2))
    top_two = np.partition(probabilities, -2, axis=2)[:, :, -2:]
    margin = top_two[:, :, 1] - top_two[:, :, 0]
    certainty = 0.60 * (1.0 - entropy) + 0.40 * margin

    lambda_quality, lambda_certainty = 0.50, 0.25
    fusion_scores = (
        lambda_quality * quality[:, None]
        + lambda_certainty * certainty
    )
    fusion_scores -= fusion_scores.max(axis=0, keepdims=True)
    weights = np.exp(fusion_scores)
    weights /= weights.sum(axis=0, keepdims=True)
    fused = np.einsum("kn,knc->nc", weights, probabilities)
    fused /= fused.sum(axis=1, keepdims=True)
    return fused, weights


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= np.clip(
        probabilities.sum(axis=1, keepdims=True), 1e-12, None
    )
    predictions = probabilities.argmax(axis=1)
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(labels, probabilities[:, 1])
    else:
        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    return {
        "AUC": float(auc),
        "ACC": float(accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "Recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the complete SAFE ensemble on the official test split."
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    spec = get_dataset_spec(args.dataset)
    artifact_dir = dataset_artifact_dir(args.artifacts, spec.key)
    score_path = artifact_dir / "branch_scores.json"
    if not score_path.exists():
        raise FileNotFoundError(
            "Screening artifacts are missing. Run train_screeners.py first."
        )
    score_payload = load_json(score_path)
    if score_payload.get("dataset") != spec.key:
        raise ValueError("The branch scores belong to a different dataset")

    _, eval_transform = classifier_transforms()
    test_dataset = MedMNISTDataset(spec, "test", transform=eval_transform)
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_order = [*BRANCHES, "baseline"]
    all_probabilities = []
    reference_labels = None
    for model_name in model_order:
        checkpoint_path = artifact_dir / "models" / f"{model_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
        checkpoint = load_torch(checkpoint_path, map_location="cpu")
        if checkpoint.get("dataset") != spec.key:
            raise ValueError(f"Checkpoint {model_name} belongs to another dataset")
        model = load_classifier(checkpoint_path, spec.num_classes, device)
        probabilities, labels = predict_probabilities(model, test_loader, device)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError("Model predictions are not aligned to the same test order")
        all_probabilities.append(probabilities)

    standardized_scores = standardize_branch_scores(
        np.asarray(score_payload["branch_scores"], dtype=np.float64)
    )
    fused_probabilities, _ = adaptive_entropy_fusion(
        np.stack(all_probabilities), standardized_scores
    )
    metrics = classification_metrics(reference_labels, fused_probabilities)
    save_json(artifact_dir / "metrics.json", metrics)

    print(f"AUC: {metrics['AUC']:.4f}")
    print(f"ACC: {metrics['ACC']:.4f}")
    print(f"F1: {metrics['F1']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")


if __name__ == "__main__":
    main()
