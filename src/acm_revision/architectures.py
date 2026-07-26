from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPModel
from torchvision.models import ResNet18_Weights, resnet18


class FusionHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float, fusion_style: str):
        super().__init__()
        self.fusion_style = fusion_style
        if fusion_style == "concat":
            in_dim = dim * 2
        elif fusion_style in ("product", "absdiff"):
            in_dim = dim
        elif fusion_style == "concat_product_absdiff":
            in_dim = dim * 4
        else:
            raise ValueError(f"Unknown fusion_style: {fusion_style}")
        self.multi_modal_projector = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        if self.fusion_style == "concat":
            fused = torch.cat([image_feat, text_feat], dim=1)
        elif self.fusion_style == "product":
            fused = image_feat * text_feat
        elif self.fusion_style == "absdiff":
            fused = torch.abs(image_feat - text_feat)
        else:
            fused = torch.cat(
                [image_feat, text_feat, image_feat * text_feat, torch.abs(image_feat - text_feat)],
                dim=1,
            )
        return self.classifier(self.multi_modal_projector(fused))


class MissingModalityMixin:
    def _init_missing(self, dim: int, missing_mode: str) -> None:
        self.missing_mode = missing_mode
        if missing_mode == "learned":
            self.missing_image_embedding = nn.Parameter(torch.zeros(1, dim))
            self.missing_text_embedding = nn.Parameter(torch.zeros(1, dim))
            nn.init.normal_(self.missing_image_embedding, std=0.02)
            nn.init.normal_(self.missing_text_embedding, std=0.02)
        elif missing_mode == "fixed":
            image_token = torch.empty(1, dim)
            text_token = torch.empty(1, dim)
            nn.init.normal_(image_token, std=0.02)
            nn.init.normal_(text_token, std=0.02)
            self.register_buffer("missing_image_embedding", image_token)
            self.register_buffer("missing_text_embedding", text_token)
        elif missing_mode in ("zero", "separate_heads"):
            self.register_buffer("missing_image_embedding", torch.zeros(1, dim))
            self.register_buffer("missing_text_embedding", torch.zeros(1, dim))
        else:
            raise ValueError(f"Unknown missing_mode: {missing_mode}")

    def _missing(self, which: str, batch_size: int, device: torch.device) -> torch.Tensor:
        token = self.missing_image_embedding if which == "image" else self.missing_text_embedding
        return token.expand(batch_size, -1).to(device)


class FlexibleCLIPNet(nn.Module, MissingModalityMixin):
    """CLIP dual encoder with configurable fusion and missing-modality handling."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        freeze_image_backbone: bool = False,
        freeze_text_backbone: bool = False,
        fusion_style: str = "concat_product_absdiff",
        fusion_stage: str = "early",
        missing_mode: str = "learned",
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        proj_dim = int(self.clip.config.projection_dim)
        self.image_proj = nn.Sequential(nn.Linear(proj_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.text_proj = nn.Sequential(nn.Linear(proj_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self._init_missing(hidden_dim, missing_mode)
        self.fusion_stage = fusion_stage
        self.fusion = FusionHead(hidden_dim, num_classes, dropout, fusion_style)
        # Stable names for layer-wise extraction and server-calibration compatibility.
        self.multi_modal_projector = self.fusion.multi_modal_projector
        self.classifier = self.fusion.classifier
        self.image_classifier = nn.Linear(hidden_dim, num_classes)
        self.text_classifier = nn.Linear(hidden_dim, num_classes)

        if freeze_image_backbone:
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = False
        if freeze_text_backbone:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False
            for p in self.clip.text_projection.parameters():
                p.requires_grad = False

    def _image_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        return self.image_norm(self.image_proj(self.clip.visual_projection(outputs.pooler_output)))

    def _text_feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_norm(self.text_proj(self.clip.text_projection(outputs.pooler_output)))

    def forward(
        self,
        image=None,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        setting: str = "both",
    ) -> torch.Tensor:
        pixel_values = pixel_values if pixel_values is not None else image
        if input_ids is not None:
            batch_size = input_ids.size(0)
            device = input_ids.device
        elif pixel_values is not None:
            batch_size = pixel_values.size(0)
            device = pixel_values.device
        else:
            raise ValueError("No input was provided.")

        use_image = setting in ("image", "image_only", "both", "multimodal", "full_multimodal")
        use_text = setting in ("text", "text_only", "both", "multimodal", "full_multimodal")
        image_feat = self._image_feature(pixel_values) if use_image else self._missing("image", batch_size, device)
        text_feat = self._text_feature(input_ids, attention_mask) if use_text else self._missing("text", batch_size, device)

        if self.fusion_stage == "late" or self.missing_mode == "separate_heads":
            if use_image and use_text:
                return 0.5 * (self.image_classifier(image_feat) + self.text_classifier(text_feat))
            if use_image:
                return self.image_classifier(image_feat)
            return self.text_classifier(text_feat)
        if self.fusion_stage != "early":
            raise ValueError(f"Unknown fusion_stage: {self.fusion_stage}")
        return self.fusion(image_feat, text_feat)


class ResNetRobertaNet(nn.Module, MissingModalityMixin):
    """Substantially different architecture: ResNet18 + RoBERTa + configurable fusion."""

    def __init__(
        self,
        text_model_name: str,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        pretrained_image: bool = True,
        freeze_image_backbone: bool = False,
        freeze_text_backbone: bool = False,
        fusion_style: str = "concat_product_absdiff",
        fusion_stage: str = "early",
        missing_mode: str = "learned",
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_image else None
        backbone = resnet18(weights=weights)
        image_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = int(self.text_encoder.config.hidden_size)
        self.image_proj = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self._init_missing(hidden_dim, missing_mode)
        self.fusion_stage = fusion_stage
        self.fusion = FusionHead(hidden_dim, num_classes, dropout, fusion_style)
        self.multi_modal_projector = self.fusion.multi_modal_projector
        self.classifier = self.fusion.classifier
        self.image_classifier = nn.Linear(hidden_dim, num_classes)
        self.text_classifier = nn.Linear(hidden_dim, num_classes)

        if freeze_image_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_text_backbone:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def _text_feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.text_norm(self.text_proj(pooled))

    def _image_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_norm(self.image_proj(self.image_encoder(pixel_values)))

    def forward(
        self,
        image=None,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        setting: str = "both",
    ) -> torch.Tensor:
        pixel_values = pixel_values if pixel_values is not None else image
        if input_ids is not None:
            batch_size = input_ids.size(0)
            device = input_ids.device
        elif pixel_values is not None:
            batch_size = pixel_values.size(0)
            device = pixel_values.device
        else:
            raise ValueError("No input was provided.")

        use_image = setting in ("image", "image_only", "both", "multimodal", "full_multimodal")
        use_text = setting in ("text", "text_only", "both", "multimodal", "full_multimodal")
        image_feat = self._image_feature(pixel_values) if use_image else self._missing("image", batch_size, device)
        text_feat = self._text_feature(input_ids, attention_mask) if use_text else self._missing("text", batch_size, device)

        if self.fusion_stage == "late" or self.missing_mode == "separate_heads":
            if use_image and use_text:
                return 0.5 * (self.image_classifier(image_feat) + self.text_classifier(text_feat))
            if use_image:
                return self.image_classifier(image_feat)
            return self.text_classifier(text_feat)
        return self.fusion(image_feat, text_feat)


def build_architecture(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    arch = model_cfg.get("architecture", "clip_dual")
    common = dict(
        num_classes=int(cfg["data"]["num_classes"]),
        hidden_dim=int(model_cfg.get("hidden_dim", model_cfg.get("projector_hidden_dim", 512))),
        dropout=float(model_cfg.get("dropout", 0.5)),
        freeze_image_backbone=bool(model_cfg.get("freeze_image_backbone", False)),
        freeze_text_backbone=bool(model_cfg.get("freeze_text_backbone", False)),
        fusion_style=model_cfg.get("fusion_style", "concat_product_absdiff"),
        fusion_stage=model_cfg.get("fusion_stage", "early"),
        missing_mode=model_cfg.get("missing_mode", "learned"),
    )
    if arch == "clip_dual":
        return FlexibleCLIPNet(model_name=model_cfg["image_model_name"], **common)
    if arch == "resnet_roberta":
        return ResNetRobertaNet(
            text_model_name=model_cfg.get("text_model_name", "roberta-base"),
            pretrained_image=bool(model_cfg.get("pretrained_image", True)),
            **common,
        )
    raise ValueError(f"Unknown architecture: {arch}")
