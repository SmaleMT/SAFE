from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import INFO
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm


SUPPORTED_DATASETS = (
    "bloodmnist",
    "breastmnist",
    "dermamnist",
    "organcmnist",
    "organsmnist",
    "pneumoniamnist",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    data_class: Any
    num_classes: int
    channels: int
    class_names: List[str]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataset_artifact_dir(root: Path, dataset: str) -> Path:
    path = root.expanduser().resolve() / dataset.lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_torch(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_dataset_spec(dataset: str) -> DatasetSpec:
    key = dataset.lower()
    if key not in SUPPORTED_DATASETS:
        choices = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose one of: {choices}")
    info = INFO[key]
    data_class = getattr(medmnist, info["python_class"])
    sample = data_class(split="train", download=True)
    labels = np.asarray(sample.labels).reshape(-1).astype(np.int64)
    label_map: Dict[str, str] = info.get("label", {})
    num_classes = max(len(label_map), int(labels.max()) + 1)
    first_image = np.asarray(sample.imgs[0])
    channels = 1 if first_image.ndim == 2 else int(first_image.shape[-1])
    class_names = [str(label_map.get(str(index), index)) for index in range(num_classes)]
    return DatasetSpec(key, data_class, num_classes, channels, class_names)


class MedMNISTDataset(Dataset):
    def __init__(self, spec: DatasetSpec, split: str, transform: Optional[Any] = None):
        self.dataset = spec.data_class(split=split, download=True)
        self.transform = transform
        self.labels = np.asarray(self.dataset.labels).reshape(-1).astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        image = self.dataset.imgs[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class MixedDataset(Dataset):
    def __init__(
        self,
        real_dataset: MedMNISTDataset,
        fake_images: torch.Tensor,
        fake_labels: torch.Tensor,
        transform: Any,
    ):
        self.real_dataset = real_dataset
        self.fake_images = fake_images.cpu()
        self.fake_labels = fake_labels.long().cpu()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.real_dataset) + len(self.fake_labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        if index < len(self.real_dataset):
            image, label = self.real_dataset[index]
            return self.transform(image), label, 0
        fake_index = index - len(self.real_dataset)
        image = self.fake_images[fake_index].float()
        image = (image * 0.5 + 0.5).clamp(0.0, 1.0)
        if image.ndim == 3 and image.shape[0] == 1:
            array = (image.squeeze(0).numpy() * 255).astype(np.uint8)
        else:
            array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return self.transform(array), int(self.fake_labels[fake_index]), 1


def convert_rgb(image: Any) -> Any:
    return image.convert("RGB")


def classifier_transforms() -> Tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Lambda(convert_rgb),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=7,
                translate=(0.02, 0.02),
                scale=(0.96, 1.04),
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Lambda(convert_rgb),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform


def gan_transform(channels: int) -> Any:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize([0.5] * channels, [0.5] * channels),
        ]
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def class_counts(spec: DatasetSpec) -> np.ndarray:
    train_dataset = MedMNISTDataset(spec, "train")
    return np.bincount(train_dataset.labels, minlength=spec.num_classes)


def spectral_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> nn.Module:
    return nn.utils.spectral_norm(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
    )


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(channels, affine=False)
        self.gamma = nn.Embedding(num_classes, channels)
        self.beta = nn.Embedding(num_classes, channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized = self.batch_norm(inputs)
        gamma = self.gamma(labels).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(labels).unsqueeze(-1).unsqueeze(-1)
        return gamma * normalized + beta


class SelfAttention(nn.Module):
    def __init__(self, channels: int, spectral: bool):
        super().__init__()
        attention_channels = max(channels // 8, 1)
        conv = spectral_conv2d if spectral else nn.Conv2d
        self.query = conv(channels, attention_channels, 1)
        self.key = conv(channels, attention_channels, 1)
        self.value = conv(channels, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        query = self.query(inputs).view(batch, -1, height * width).transpose(1, 2)
        key = self.key(inputs).view(batch, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(inputs).view(batch, -1, height * width)
        output = torch.bmm(value, attention.transpose(1, 2))
        return inputs + self.scale * output.view(batch, channels, height, width)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.xavier_uniform_(self.conv1.weight, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.conv2.weight, gain=math.sqrt(2.0))

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        hidden = F.interpolate(inputs, scale_factor=2, mode="nearest")
        hidden = F.relu(self.norm1(self.conv1(hidden), labels))
        hidden = self.norm2(self.conv2(hidden), labels)
        return F.relu(hidden + self.skip(residual))


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = spectral_conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = spectral_conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            spectral_conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels or downsample
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(inputs)
        hidden = F.relu(self.conv1(hidden))
        hidden = self.conv2(hidden)
        shortcut = inputs
        if self.downsample:
            hidden = F.avg_pool2d(hidden, 2)
            shortcut = F.avg_pool2d(shortcut, 2)
        return hidden + self.skip(shortcut)


class DiscriminatorFirstBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = spectral_conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = spectral_conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = spectral_conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(inputs))
        hidden = F.avg_pool2d(self.conv2(hidden), 2)
        return hidden + self.skip(F.avg_pool2d(inputs, 2))


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        base_channels: int = 128,
        output_channels: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_channels = base_channels * 4
        self.project = nn.Linear(latent_dim, self.initial_channels * 7 * 7)
        self.block1 = GeneratorBlock(
            self.initial_channels, base_channels * 2, num_classes
        )
        self.attention = SelfAttention(base_channels * 2, spectral=False)
        self.block2 = GeneratorBlock(base_channels * 2, base_channels, num_classes)
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output_conv = nn.Conv2d(base_channels, output_channels, 3, padding=1)
        nn.init.xavier_uniform_(self.output_conv.weight)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hidden = self.project(noise)
        hidden = hidden.view(noise.shape[0], self.initial_channels, 7, 7)
        hidden = self.block1(hidden, labels)
        hidden = self.attention(hidden)
        hidden = self.block2(hidden, labels)
        hidden = F.relu(self.output_norm(hidden))
        return torch.tanh(self.output_conv(hidden))


class ProjectionDiscriminator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        base_channels: int = 128,
    ):
        super().__init__()
        self.block1 = DiscriminatorFirstBlock(input_channels, base_channels)
        self.attention = SelfAttention(base_channels, spectral=True)
        self.block2 = DiscriminatorBlock(base_channels, base_channels * 2)
        self.block3 = DiscriminatorBlock(
            base_channels * 2, base_channels * 4, downsample=False
        )
        self.output = nn.utils.spectral_norm(nn.Linear(base_channels * 4, 1))
        self.class_embedding = nn.utils.spectral_norm(
            nn.Embedding(num_classes, base_channels * 4)
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hidden = self.block1(images)
        hidden = self.attention(hidden)
        hidden = self.block2(hidden)
        hidden = F.relu(self.block3(hidden))
        hidden = hidden.sum(dim=(2, 3))
        unconditional = self.output(hidden)
        projection = (hidden * self.class_embedding(labels)).sum(dim=1, keepdim=True)
        return (unconditional + projection).reshape(-1)


def random_translation(images: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    batch, _, height, width = images.shape
    shift_x = max(1, int(height * ratio + 0.5))
    shift_y = max(1, int(width * ratio + 0.5))
    translation_x = torch.randint(
        -shift_x, shift_x + 1, (batch, 1, 1), device=images.device
    )
    translation_y = torch.randint(
        -shift_y, shift_y + 1, (batch, 1, 1), device=images.device
    )
    grid_batch, grid_height, grid_width = torch.meshgrid(
        torch.arange(batch, device=images.device),
        torch.arange(height, device=images.device),
        torch.arange(width, device=images.device),
        indexing="ij",
    )
    grid_height = torch.clamp(grid_height + translation_x, 0, height - 1)
    grid_width = torch.clamp(grid_width + translation_y, 0, width - 1)
    output = images.permute(0, 2, 3, 1)[
        grid_batch, grid_height, grid_width, :
    ]
    return output.permute(0, 3, 1, 2).contiguous()


def random_cutout(images: torch.Tensor, ratio: float = 0.15) -> torch.Tensor:
    batch, _, height, width = images.shape
    size = max(1, int(height * ratio + 0.5))
    offset_x = torch.randint(
        0, max(1, height + 1 - size), (batch, 1, 1), device=images.device
    )
    offset_y = torch.randint(
        0, max(1, width + 1 - size), (batch, 1, 1), device=images.device
    )
    _, grid_height, grid_width = torch.meshgrid(
        torch.arange(batch, device=images.device),
        torch.arange(height, device=images.device),
        torch.arange(width, device=images.device),
        indexing="ij",
    )
    cutout = (
        (grid_height >= offset_x)
        & (grid_height < offset_x + size)
        & (grid_width >= offset_y)
        & (grid_width < offset_y + size)
    )
    return images * (~cutout).unsqueeze(1).to(images.dtype)


def diff_augment(images: torch.Tensor) -> torch.Tensor:
    return random_cutout(random_translation(images))


def discriminator_hinge_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SAFE conditional GAN and export its candidate pool."
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=4e-4)
    parser.add_argument("--discriminator-steps", type=int, default=2)
    parser.add_argument("--fid-interval", type=int, default=10)
    parser.add_argument("--fid-samples", type=int, default=2048)
    parser.add_argument("--pool-factor", type=int, default=10)
    return parser.parse_args()


def frechet_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    difference = mean_a - mean_b
    covariance_mean, _ = linalg.sqrtm(
        covariance_a.dot(covariance_b), disp=False
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_a.shape[0]) * epsilon
        covariance_mean = linalg.sqrtm(
            (covariance_a + offset).dot(covariance_b + offset)
        )
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    value = (
        difference.dot(difference)
        + np.trace(covariance_a)
        + np.trace(covariance_b)
        - 2.0 * np.trace(covariance_mean)
    )
    return float(value)


def state_dict_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def inception_input(images: torch.Tensor) -> torch.Tensor:
    images = images.float() * 0.5 + 0.5
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)


@torch.no_grad()
def inception_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    features: List[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = inception_input(images[start : start + batch_size]).to(device)
        features.append(model(batch).cpu().numpy())
    return np.concatenate(features)


def fixed_real_sample(
    dataset: MedMNISTDataset,
    sample_count: int,
) -> torch.Tensor:
    replace = len(dataset) < sample_count
    indices = np.random.choice(len(dataset), size=sample_count, replace=replace)
    return torch.stack([dataset[int(index)][0] for index in indices])


@torch.no_grad()
def evaluate_fid(
    generator: ConditionalGenerator,
    noise: torch.Tensor,
    labels: torch.Tensor,
    real_statistics: Tuple[np.ndarray, np.ndarray],
    inception: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    generator.eval()
    generated = []
    for start in range(0, len(labels), batch_size):
        generated.append(
            generator(
                noise[start : start + batch_size],
                labels[start : start + batch_size],
            ).cpu()
        )
    fake_features = inception_features(
        inception, torch.cat(generated), device, batch_size
    )
    fake_mean = fake_features.mean(axis=0)
    fake_covariance = np.cov(fake_features, rowvar=False)
    return frechet_distance(
        real_statistics[0],
        real_statistics[1],
        fake_mean,
        fake_covariance,
    )


@torch.no_grad()
def export_candidate_pool(
    generator: ConditionalGenerator,
    deficits: np.ndarray,
    pool_factor: int,
    latent_dim: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    generator.eval()
    pool: Dict[int, Dict[str, torch.Tensor]] = {}
    for class_index, deficit in enumerate(deficits.tolist()):
        count = int(deficit * pool_factor)
        if count <= 0:
            continue
        parts = []
        for start in range(0, count, 128):
            current = min(128, count - start)
            noise = torch.randn(current, latent_dim, device=device)
            labels = torch.full(
                (current,), class_index, dtype=torch.long, device=device
            )
            parts.append(generator(noise, labels).cpu().half())
        images = torch.cat(parts)
        pool[class_index] = {
            "imgs": images,
            "lbls": torch.full((count,), class_index, dtype=torch.long),
        }
    return pool


def main() -> None:
    args = parse_args()
    device = get_device()
    spec = get_dataset_spec(args.dataset)
    artifact_dir = dataset_artifact_dir(args.artifacts, spec.key)
    gan_dir = artifact_dir / "gan"
    gan_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MedMNISTDataset(
        spec, "train", transform=gan_transform(spec.channels)
    )
    counts = class_counts(spec)
    deficits = counts.max() - counts
    labels_array = train_dataset.labels
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    generator = ConditionalGenerator(
        args.latent_dim,
        spec.num_classes,
        args.base_channels,
        spec.channels,
    ).to(device)
    discriminator = ProjectionDiscriminator(
        spec.num_classes,
        spec.channels,
        args.base_channels,
    ).to(device)
    generator_optimizer = Adam(
        generator.parameters(), lr=args.generator_lr, betas=(0.0, 0.999)
    )
    discriminator_optimizer = Adam(
        discriminator.parameters(),
        lr=args.discriminator_lr,
        betas=(0.0, 0.999),
    )

    def decay(epoch: int) -> float:
        midpoint = args.epochs // 2
        if epoch <= midpoint:
            return 1.0
        return max(0.0, 1.0 - (epoch - midpoint) / max(args.epochs - midpoint, 1))

    generator_scheduler = LambdaLR(generator_optimizer, decay)
    discriminator_scheduler = LambdaLR(discriminator_optimizer, decay)

    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception.fc = torch.nn.Identity()
    inception = inception.to(device).eval()
    real_images = fixed_real_sample(train_dataset, args.fid_samples)
    real_features = inception_features(inception, real_images, device)
    real_statistics = (
        real_features.mean(axis=0),
        np.cov(real_features, rowvar=False),
    )
    fid_labels_np = np.random.choice(
        labels_array, size=args.fid_samples, replace=True
    )
    fid_labels = torch.tensor(fid_labels_np, dtype=torch.long, device=device)
    fid_noise = torch.randn(args.fid_samples, args.latent_dim, device=device)

    best_fid = float("inf")
    best_epoch = 0
    best_state = state_dict_cpu(generator)
    label_tensor = torch.tensor(labels_array, dtype=torch.long, device=device)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        iterator = iter(train_loader)
        progress = tqdm(range(len(train_loader)), desc=f"epoch {epoch:03d}/{args.epochs:03d}")
        for _ in progress:
            for _ in range(args.discriminator_steps):
                try:
                    real_images_batch, real_labels = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    real_images_batch, real_labels = next(iterator)
                real_images_batch = real_images_batch.to(device)
                real_labels = real_labels.long().to(device)
                noise = torch.randn(
                    real_images_batch.shape[0], args.latent_dim, device=device
                )
                with torch.no_grad():
                    fake_images = generator(noise, real_labels)
                real_logits = discriminator(
                    diff_augment(real_images_batch), real_labels
                )
                fake_logits = discriminator(diff_augment(fake_images), real_labels)
                discriminator_loss = discriminator_hinge_loss(
                    real_logits, fake_logits
                )
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss.backward()
                discriminator_optimizer.step()

            sample_indices = torch.randint(
                0, len(label_tensor), (args.batch_size,), device=device
            )
            generated_labels = label_tensor[sample_indices]
            noise = torch.randn(args.batch_size, args.latent_dim, device=device)
            generated_images = generator(noise, generated_labels)
            generator_loss = generator_hinge_loss(
                discriminator(diff_augment(generated_images), generated_labels)
            )
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            generator_optimizer.step()
            progress.set_postfix(
                D=f"{discriminator_loss.item():.3f}",
                G=f"{generator_loss.item():.3f}",
            )

        generator_scheduler.step()
        discriminator_scheduler.step()
        if epoch % args.fid_interval == 0 or epoch == args.epochs:
            fid = evaluate_fid(
                generator,
                fid_noise,
                fid_labels,
                real_statistics,
                inception,
                device,
            )
            print(f"epoch={epoch} FID={fid:.4f}")
            if fid < best_fid:
                best_fid = fid
                best_epoch = epoch
                best_state = state_dict_cpu(generator)
                torch.save(
                    {
                        "dataset": spec.key,
                        "epoch": best_epoch,
                        "fid": best_fid,
                        "latent_dim": args.latent_dim,
                        "base_channels": args.base_channels,
                        "channels": spec.channels,
                        "num_classes": spec.num_classes,
                        "state_dict": best_state,
                    },
                    gan_dir / "generator.pt",
                )

    generator.load_state_dict(best_state)
    candidate_pool = export_candidate_pool(
        generator,
        deficits,
        args.pool_factor,
        args.latent_dim,
        device,
    )
    torch.save(candidate_pool, gan_dir / "candidate_pool.pt")
    save_json(
        gan_dir / "metadata.json",
        {
            "dataset": spec.key,
            "best_epoch": best_epoch,
            "best_fid": best_fid,
            "class_counts": counts.tolist(),
            "class_deficits": deficits.tolist(),
            "pool_factor": args.pool_factor,
        },
    )
    print(f"saved candidate pool to {gan_dir / 'candidate_pool.pt'}")


if __name__ == "__main__":
    main()
