"""Frozen feature backbones.

Default is DINOv2 ViT-B/14 (plan.md line 86): the team already has it, the
checkpoint is cached at $TORCH_HOME, and DINOSAUR (Seitzer et al., ICLR 2023)
showed frozen self-supervised ViT features are precisely what let slots discover
objects in real images.

Patch tokens are kept rather than pooled -- at 224px input with patch size 14
that is a 16x16 grid per slice, which is the spatial resolution the localisation
evaluation needs. Pooling to a CLS vector would make Dice against a lesion mask
meaningless.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DINOV2_VARIANTS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# ImageNet statistics -- DINOv2 was trained with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2PatchExtractor(nn.Module):
    """Frozen DINOv2, returning normalised patch tokens ``[B, P, D]``."""

    def __init__(self, variant: str = "dinov2_vitb14", image_size: int = 224):
        super().__init__()
        if variant not in DINOV2_VARIANTS:
            raise ValueError(
                f"{variant!r} not one of {sorted(DINOV2_VARIANTS)}"
            )
        if image_size % 14 != 0:
            raise ValueError(
                f"image_size must be a multiple of the patch size 14, got {image_size}"
            )
        self.model = torch.hub.load("facebookresearch/dinov2", variant)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.variant = variant
        self.dim = DINOV2_VARIANTS[variant]
        self.image_size = image_size
        self.grid = image_size // 14
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, 3, H, W)`` in [0, 1]. Returns ``(B, grid*grid, dim)``."""
        x = (x - self.mean) / self.std
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]


class DenseNet121Extractor(nn.Module):
    """DenseNet-121 feature map as a patch grid.

    ImageNet weights by default. RadImageNet weights (plan.md line 86) load via
    ``weights_path`` once obtained -- they are gated behind a Google Drive request
    form, so this stays optional rather than blocking the ablation.
    """

    def __init__(self, image_size: int = 224, weights_path: str | None = None):
        super().__init__()
        import torchvision

        net = torchvision.models.densenet121(
            weights=None if weights_path else torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        )
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            state = state.get("state_dict", state)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = net.load_state_dict(state, strict=False)
            if len(missing) > 50:
                raise RuntimeError(
                    f"{weights_path} does not look like DenseNet-121 weights "
                    f"({len(missing)} missing keys)"
                )
        self.features = net.features
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.dim = 1024
        self.image_size = image_size
        self.grid = image_size // 32  # DenseNet-121 total stride
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        fmap = self.features(x)  # B, 1024, g, g
        return fmap.flatten(2).transpose(1, 2)  # B, g*g, 1024


def build_backbone(name: str, image_size: int = 224, **kw) -> nn.Module:
    if name.startswith("dinov2"):
        return DINOv2PatchExtractor(name, image_size=image_size)
    if name == "densenet121":
        return DenseNet121Extractor(
            image_size=image_size, weights_path=kw.get("weights_path")
        )
    raise ValueError(f"unknown backbone {name!r}")
