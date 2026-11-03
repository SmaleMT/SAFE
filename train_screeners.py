from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from train_gan import (
    SUPPORTED_DATASETS,
    MedMNISTDataset,
    MixedDataset,
    class_counts,
    classifier_transforms,
    gan_transform,
    get_dataset_spec,
    make_loader,
    dataset_artifact_dir,
    get_device,
    load_json,
    load_torch,
    save_json,
)


BRANCHES = ("chre", "chre_fa", "chre_mmr", "chre_fa_mmr")


@dataclass(frozen=True)
class ScreeningConfig:
    neighbors: int = 5
    contribution_balance: float = 0.55
    contribution_weight: float = 0.25
    margin_weight: float = 0.25
    density_weight: float = 0.25
    teacher_weight: float = 0.25
    mmr_pool_factor: int = 4
    diversity_weight: float = 0.35
    augmentation_ratio: float = 1.0


def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_feature_extractor(
    num_classes: int,
    aligned_state: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    model = build_classifier(num_classes, pretrained=True)
    if aligned_state is not None:
        model.load_state_dict(aligned_state)
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def to_imagenet_input(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    images = images * 0.5 + 0.5
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    for batch in loader:
        images, batch_labels = batch[:2]
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


@torch.no_grad()
def validation_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        images, labels = batch[:2]
        predictions = model(images.to(device, non_blocking=True)).argmax(dim=1)
        labels = labels.to(device)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_path: Path,
    dataset: str,
    synthetic_weight: Optional[float] = None,
) -> nn.Module:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03, reduction="none")
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    best_accuracy = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}/{epochs:03d}")
        for batch in progress:
            images, labels = batch[:2]
            source = batch[2] if len(batch) == 3 else None
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(images), labels)
            if source is not None and synthetic_weight is not None:
                source = source.to(device)
                weights = torch.where(
                    source == 1,
                    torch.full_like(losses, synthetic_weight),
                    torch.ones_like(losses),
                )
                loss = (losses * weights).mean()
            else:
                loss = losses.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * labels.numel()
            sample_count += int(labels.numel())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        accuracy = validation_accuracy(model, validation_loader, device)
        mean_loss = running_loss / max(sample_count, 1)
        print(f"validation ACC={accuracy:.4f} train_loss={mean_loss:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "dataset": dataset,
                    "epoch": epoch,
                    "validation_accuracy": best_accuracy,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                },
                checkpoint_path,
            )

    checkpoint = load_torch(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_classifier(
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    checkpoint: Dict[str, Any] = load_torch(checkpoint_path, map_location=device)
    model = build_classifier(num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def branch_feature_space(branch: str) -> str:
    if branch not in BRANCHES:
        raise ValueError(f"Unknown screening branch: {branch}")
    return "medical" if "fa" in branch else "generic"


def uses_mmr(branch: str) -> bool:
    return branch.endswith("mmr")


def minmax(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if values.numel() == 0:
        return values
    return (values - values.min()) / (values.max() - values.min() + epsilon)


def binary_entropy(same_count: int, different_count: int) -> float:
    total = same_count + different_count
    if total == 0:
        return 0.0
    result = 0.0
    for count in (same_count, different_count):
        if count:
            probability = count / total
            result -= probability * math.log(probability)
    return result


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    backbone.eval()
    features = backbone(to_imagenet_input(images.to(device)))
    return F.normalize(features, p=2, dim=1)


@torch.no_grad()
def collect_features(
    backbone: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    features = []
    labels = []
    for images, batch_labels in tqdm(loader, desc="extract real features"):
        features.append(extract_features(backbone, images, device).cpu())
        labels.append(batch_labels.long())
    return torch.cat(features).to(device), torch.cat(labels).to(device)


@torch.no_grad()
def reverse_knn_radius(
    features: torch.Tensor,
    neighbors: int,
    chunk_size: int = 1024,
) -> torch.Tensor:
    count = features.shape[0]
    k = min(neighbors, max(count - 1, 1))
    radii = []
    for start in tqdm(range(0, count, chunk_size), desc="real-neighbor radii"):
        end = min(start + chunk_size, count)
        distances = 1.0 - features[start:end] @ features.T
        local = torch.arange(end - start, device=features.device)
        global_indices = torch.arange(start, end, device=features.device)
        distances[local, global_indices] = float("inf")
        radii.append(torch.topk(distances, k, largest=False).values[:, -1])
    return torch.cat(radii)


@torch.no_grad()
def score_candidates(
    images: torch.Tensor,
    target_class: int,
    real_features: torch.Tensor,
    real_labels: torch.Tensor,
    real_radii: torch.Tensor,
    backbone: nn.Module,
    teacher: nn.Module,
    num_classes: int,
    config: ScreeningConfig,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    same_features = real_features[real_labels == target_class]
    different_features = real_features[real_labels != target_class]
    maximum_entropy = -math.log(1.0 / num_classes)
    feature_parts: List[torch.Tensor] = []
    contribution_parts: List[torch.Tensor] = []
    margin_parts: List[torch.Tensor] = []
    density_parts: List[torch.Tensor] = []
    confidence_parts: List[torch.Tensor] = []

    for start in tqdm(
        range(0, len(images), batch_size),
        desc=f"score class {target_class}",
    ):
        batch = images[start : start + batch_size].to(device).float()
        features = extract_features(backbone, batch, device)
        feature_parts.append(features.cpu())
        all_distances = 1.0 - features @ real_features.T
        k = min(config.neighbors, all_distances.shape[1])
        neighbor_indices = torch.topk(all_distances, k, largest=False).indices
        neighbor_labels = real_labels[neighbor_indices]
        same_distances = 1.0 - features @ same_features.T
        same_nearest = torch.topk(
            same_distances,
            min(config.neighbors, same_distances.shape[1]),
            largest=False,
        ).values
        nearest_same = same_nearest[:, 0]
        density = torch.exp(-same_nearest.mean(dim=1))
        if different_features.numel():
            nearest_different = (1.0 - features @ different_features.T).min(dim=1).values
        else:
            nearest_different = torch.full_like(nearest_same, 2.0)
        margin = nearest_different - nearest_same
        teacher.eval()
        confidence = torch.softmax(teacher(to_imagenet_input(batch)), dim=1)[
            :, target_class
        ]

        contributions = []
        for row in range(features.shape[0]):
            same_count = int((neighbor_labels[row] == target_class).sum().item())
            different_count = k - same_count
            local_entropy = binary_entropy(same_count, different_count)
            information = (
                local_entropy
                if same_count >= k / 2
                else 2 * maximum_entropy - local_entropy
            )
            reverse_mask = all_distances[row] <= real_radii
            reverse_labels = real_labels[reverse_mask]
            reverse_same = int((reverse_labels == target_class).sum().item())
            reverse_different = int(reverse_labels.numel()) - reverse_same
            reverse_entropy = binary_entropy(reverse_same, reverse_different)
            noise = (
                reverse_entropy
                if reverse_same >= reverse_labels.numel() / 2
                else 2 * maximum_entropy - reverse_entropy
            )
            contributions.append(
                (1.0 - config.contribution_balance) * information
                - config.contribution_balance * noise
            )

        contribution_parts.append(torch.tensor(contributions))
        margin_parts.append(margin.cpu())
        density_parts.append(density.cpu())
        confidence_parts.append(confidence.cpu())

    features = torch.cat(feature_parts)
    contribution = torch.cat(contribution_parts).float()
    margin = torch.cat(margin_parts).float()
    density = torch.cat(density_parts).float()
    confidence = torch.cat(confidence_parts).float().clamp(0.0, 1.0)
    score = (
        config.contribution_weight * minmax(contribution)
        + config.margin_weight * minmax(margin)
        + config.density_weight * minmax(density)
        + config.teacher_weight * confidence
    )
    return {
        "features": features,
        "contribution": contribution,
        "margin": margin,
        "density": density,
        "confidence": confidence,
        "score": score,
    }


def select_top_scores(score_pack: Dict[str, torch.Tensor], count: int) -> List[int]:
    count = min(count, score_pack["score"].numel())
    if count <= 0:
        return []
    return torch.topk(score_pack["score"], count, largest=True).indices.tolist()


def select_with_mmr(
    score_pack: Dict[str, torch.Tensor],
    count: int,
    config: ScreeningConfig,
    device: torch.device,
) -> List[int]:
    scores = score_pack["score"]
    count = min(count, scores.numel())
    if count <= 0:
        return []
    pool_size = min(scores.numel(), count * config.mmr_pool_factor)
    pool_indices = torch.topk(scores, pool_size, largest=True).indices
    relevance = minmax(scores[pool_indices]).to(device)
    features = score_pack["features"][pool_indices].to(device)
    available = torch.ones(pool_size, dtype=torch.bool, device=device)
    first = int(relevance.argmax().item())
    selected = [first]
    available[first] = False
    nearest_selected_distance = torch.full(
        (pool_size,), float("inf"), device=device
    )
    while len(selected) < count:
        last_feature = features[selected[-1]].unsqueeze(0)
        distances = 1.0 - features[available] @ last_feature.T
        nearest_selected_distance[available] = torch.minimum(
            nearest_selected_distance[available], distances.squeeze(1)
        )
        diversity = minmax(nearest_selected_distance[available])
        combined = (
            (1.0 - config.diversity_weight) * relevance[available]
            + config.diversity_weight * diversity
        )
        available_indices = torch.nonzero(available).squeeze(1)
        next_index = int(available_indices[combined.argmax()].item())
        selected.append(next_index)
        available[next_index] = False
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    return pool_indices[selected_tensor].tolist()


def select_branch_samples(
    branch: str,
    candidate_pool: Dict[Any, Dict[str, torch.Tensor]],
    deficits: np.ndarray,
    score_cache: Dict[Tuple[str, int], Dict[str, torch.Tensor]],
    config: ScreeningConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, Any]]:
    space = branch_feature_space(branch)
    selected_images = []
    selected_labels = []
    class_means = []
    class_report: Dict[str, Any] = {}
    for class_index, deficit in enumerate(deficits.tolist()):
        count = int(round(deficit * config.augmentation_ratio))
        if count <= 0:
            continue
        pool_entry = candidate_pool.get(class_index, candidate_pool.get(str(class_index)))
        if pool_entry is None:
            raise KeyError(f"Candidate pool has no entry for class {class_index}")
        score_pack = score_cache[(space, class_index)]
        if uses_mmr(branch):
            indices = select_with_mmr(score_pack, count, config, device)
        else:
            indices = select_top_scores(score_pack, count)
        index_tensor = torch.tensor(indices, dtype=torch.long)
        selected_images.append(pool_entry["imgs"][index_tensor])
        selected_labels.append(
            torch.full((len(indices),), class_index, dtype=torch.long)
        )
        mean_score = float(score_pack["score"][index_tensor].mean().item())
        class_means.append(mean_score)
        class_report[str(class_index)] = {
            "count": len(indices),
            "mean_screening_score": mean_score,
            "indices": indices,
        }
    if not selected_images:
        raise RuntimeError(f"Branch '{branch}' did not select any samples")
    branch_score = float(np.mean(class_means))
    return (
        torch.cat(selected_images),
        torch.cat(selected_labels),
        branch_score,
        {
            "branch": branch,
            "feature_space": space,
            "branch_score": branch_score,
            "classes": class_report,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen generated samples and train the five SAFE classifiers."
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--fa-epochs", type=int, default=10)
    parser.add_argument("--classifier-epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--synthetic-weight", type=float, default=0.45)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--contribution-balance", type=float, default=0.55)
    parser.add_argument("--mmr-pool-factor", type=int, default=4)
    parser.add_argument("--diversity-weight", type=float, default=0.35)
    return parser.parse_args()


def real_data_loaders(
    spec: Any,
    train_transform: Any,
    eval_transform: Any,
    batch_size: int,
    num_workers: int,
) -> Tuple[Any, Any]:
    train_dataset = MedMNISTDataset(spec, "train", train_transform)
    validation_dataset = MedMNISTDataset(spec, "val", eval_transform)
    return (
        make_loader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        make_loader(
            validation_dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )


def train_real_classifier(
    name: str,
    epochs: int,
    spec: Any,
    train_transform: Any,
    eval_transform: Any,
    args: argparse.Namespace,
    model_dir: Path,
    device: torch.device,
) -> nn.Module:
    train_loader, validation_loader = real_data_loaders(
        spec,
        train_transform,
        eval_transform,
        args.batch_size,
        args.num_workers,
    )
    print(f"training {name}")
    return train_classifier(
        build_classifier(spec.num_classes, pretrained=True),
        train_loader,
        validation_loader,
        device,
        epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_path=model_dir / f"{name}.pt",
        dataset=spec.key,
    )


def validate_gan_artifacts(
    artifact_dir: Path,
    dataset: str,
) -> Dict[int, Dict[str, torch.Tensor]]:
    metadata_path = artifact_dir / "gan" / "metadata.json"
    pool_path = artifact_dir / "gan" / "candidate_pool.pt"
    if not metadata_path.exists() or not pool_path.exists():
        raise FileNotFoundError(
            "GAN artifacts are missing. Run train_gan.py for this dataset first."
        )
    metadata = load_json(metadata_path)
    if metadata.get("dataset") != dataset:
        raise ValueError("The candidate pool belongs to a different dataset")
    return load_torch(pool_path, map_location="cpu")


def main() -> None:
    args = parse_args()
    device = get_device()
    spec = get_dataset_spec(args.dataset)
    artifact_dir = dataset_artifact_dir(args.artifacts, spec.key)
    model_dir = artifact_dir / "models"
    selection_dir = artifact_dir / "selections"
    model_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)
    candidate_pool = validate_gan_artifacts(artifact_dir, spec.key)
    counts = class_counts(spec)
    deficits = counts.max() - counts
    train_transform, eval_transform = classifier_transforms()
    feature_transform = gan_transform(spec.channels)
    config = ScreeningConfig(
        neighbors=args.neighbors,
        contribution_balance=args.contribution_balance,
        mmr_pool_factor=args.mmr_pool_factor,
        diversity_weight=args.diversity_weight,
    )

    feature_aligner = train_real_classifier(
        "feature_aligner",
        args.fa_epochs,
        spec,
        train_transform,
        eval_transform,
        args,
        model_dir,
        device,
    )
    teacher = train_real_classifier(
        "teacher",
        args.teacher_epochs,
        spec,
        train_transform,
        eval_transform,
        args,
        model_dir,
        device,
    ).eval()

    backbones = {
        "generic": build_feature_extractor(spec.num_classes).to(device).eval(),
        "medical": build_feature_extractor(
            spec.num_classes, aligned_state=feature_aligner.state_dict()
        ).to(device).eval(),
    }
    feature_dataset = MedMNISTDataset(
        spec, "train", transform=feature_transform
    )
    feature_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for space, backbone in backbones.items():
        real_features, real_labels = collect_features(
            backbone,
            feature_dataset,
            device,
            batch_size=128,
            num_workers=args.num_workers,
        )
        radii = reverse_knn_radius(real_features, config.neighbors)
        feature_cache[space] = (real_features, real_labels, radii)

    score_cache: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
    for space, backbone in backbones.items():
        real_features, real_labels, radii = feature_cache[space]
        for class_index, deficit in enumerate(deficits.tolist()):
            if deficit <= 0:
                continue
            pool_entry = candidate_pool.get(
                class_index, candidate_pool.get(str(class_index))
            )
            if pool_entry is None:
                raise KeyError(f"Candidate pool has no entry for class {class_index}")
            score_cache[(space, class_index)] = score_candidates(
                pool_entry["imgs"],
                class_index,
                real_features,
                real_labels,
                radii,
                backbone,
                teacher,
                spec.num_classes,
                config,
                device,
            )

    raw_train_dataset = MedMNISTDataset(spec, "train", transform=None)
    validation_dataset = MedMNISTDataset(spec, "val", transform=eval_transform)
    branch_scores = []
    branch_metadata: Dict[str, Any] = {}

    for branch in BRANCHES:
        print(f"training {branch}")
        fake_images, fake_labels, branch_score, report = select_branch_samples(
            branch,
            candidate_pool,
            deficits,
            score_cache,
            config,
            device,
        )
        save_json(selection_dir / f"{branch}.json", report)
        mixed_dataset = MixedDataset(
            raw_train_dataset,
            fake_images,
            fake_labels,
            train_transform,
        )
        train_loader = make_loader(
            mixed_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        validation_loader = make_loader(
            validation_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        train_classifier(
            build_classifier(spec.num_classes, pretrained=True),
            train_loader,
            validation_loader,
            device,
            epochs=args.classifier_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            checkpoint_path=model_dir / f"{branch}.pt",
            dataset=spec.key,
            synthetic_weight=args.synthetic_weight,
        )
        branch_scores.append(branch_score)
        branch_metadata[branch] = {
            "feature_space": branch_feature_space(branch),
            "score": branch_score,
            "selected_samples": int(len(fake_labels)),
        }

    baseline_train_loader, baseline_validation_loader = real_data_loaders(
        spec,
        train_transform,
        eval_transform,
        args.batch_size,
        args.num_workers,
    )
    print("training baseline")
    train_classifier(
        build_classifier(spec.num_classes, pretrained=True),
        baseline_train_loader,
        baseline_validation_loader,
        device,
        epochs=args.classifier_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_path=model_dir / "baseline.pt",
        dataset=spec.key,
    )

    save_json(
        artifact_dir / "branch_scores.json",
        {
            "dataset": spec.key,
            "model_order": [*BRANCHES, "baseline"],
            "branch_scores": branch_scores,
            "baseline_standardized_score": 0.0,
            "branches": branch_metadata,
        },
    )
    print(f"saved models to {model_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
