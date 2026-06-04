# ============================================================
# Model: CLIP-ViT + RoBERTa for MVSA 4-Class Classification
# ============================================================

import torch
import torch.nn as nn

from transformers import AutoModel, CLIPVisionModel


class StrongMultimodalNet(nn.Module):
    """
    Strong multimodal model:
        Image encoder: CLIP-ViT
        Text encoder: RoBERTa / BERT
        Fusion: MLP projector
        Classifier: 4-class sentiment classification
    """

    def __init__(
        self,
        text_model_name="roberta-base",
        image_model_name="openai/clip-vit-base-patch32",
        num_classes=4,
        image_hidden_dim=256,
        text_hidden_dim=256,
        projector_hidden_dim=256,
        dropout=0.3,
        freeze_image_backbone=True,
        freeze_text_backbone=True,
        pretrained_image=True,  # kept for old config compatibility, not used by CLIP
    ):
        super().__init__()

        # -------------------------
        # Image encoder: CLIP-ViT
        # -------------------------
        self.image_backbone = CLIPVisionModel.from_pretrained(image_model_name)
        clip_hidden = self.image_backbone.config.hidden_size
        self.image_proj = nn.Linear(clip_hidden, image_hidden_dim)

        if freeze_image_backbone:
            for p in self.image_backbone.parameters():
                p.requires_grad = False

        # -------------------------
        # Text encoder: RoBERTa / BERT
        # -------------------------
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        text_backbone_hidden = self.text_backbone.config.hidden_size
        self.text_proj = nn.Linear(text_backbone_hidden, text_hidden_dim)

        if freeze_text_backbone:
            for p in self.text_backbone.parameters():
                p.requires_grad = False

        # -------------------------
        # Fusion + classifier
        # -------------------------
        fusion_input_dim = image_hidden_dim + text_hidden_dim

        self.multi_modal_projector = nn.Sequential(
            nn.Linear(fusion_input_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(projector_hidden_dim, num_classes)

    def forward(
        self,
        image=None,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        setting="both",
    ):
        """
        Compatible with old training code:
            model(image, input_ids, attention_mask)

        Also compatible with new training code:
            model(pixel_values=pixel_values, input_ids=..., attention_mask=...)

        setting:
            image_only / image
            text_only / text
            both / multimodal / modality_exclusive
        """

        # Old code passes image as first positional argument.
        # For CLIP, this image tensor is actually pixel_values.
        if pixel_values is None:
            pixel_values = image

        if input_ids is not None:
            batch_size = input_ids.size(0)
            device = input_ids.device
        elif pixel_values is not None:
            batch_size = pixel_values.size(0)
            device = pixel_values.device
        else:
            raise ValueError("Either input_ids or pixel_values/image must be provided.")

        image_feat = torch.zeros(
            batch_size,
            self.image_proj.out_features,
            device=device,
        )

        text_feat = torch.zeros(
            batch_size,
            self.text_proj.out_features,
            device=device,
        )

        use_image = setting in [
            "image",
            "image_only",
            "both",
            "multimodal",
            "modality_exclusive",
        ]

        use_text = setting in [
            "text",
            "text_only",
            "both",
            "multimodal",
            "modality_exclusive",
        ]

        # -------------------------
        # Image branch
        # -------------------------
        if use_image and pixel_values is not None:
            image_outputs = self.image_backbone(pixel_values=pixel_values)

            # CLIPVisionModel provides pooler_output: [batch, hidden_size]
            image_cls = image_outputs.pooler_output
            image_feat = self.image_proj(image_cls)

        # -------------------------
        # Text branch
        # -------------------------
        if use_text and input_ids is not None and attention_mask is not None:
            text_outputs = self.text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Works for BERT, RoBERTa, DistilBERT
            text_cls = text_outputs.last_hidden_state[:, 0, :]
            text_feat = self.text_proj(text_cls)

        # -------------------------
        # Fusion
        # -------------------------
        fused = torch.cat([image_feat, text_feat], dim=1)

        h = self.multi_modal_projector(fused)
        h = self.dropout(h)

        logits = self.classifier(h)

        return logits


def build_model(args):
    """
    Build model from args/config.
    """

    image_model_name = getattr(
        args,
        "image_model_name",
        "openai/clip-vit-base-patch32",
    )

    pretrained_image = getattr(args, "pretrained_image", True)

    model = StrongMultimodalNet(
        text_model_name=args.text_model_name,
        image_model_name=image_model_name,
        num_classes=args.num_classes,
        image_hidden_dim=args.image_hidden_dim,
        text_hidden_dim=args.text_hidden_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        dropout=args.dropout,
        freeze_image_backbone=args.freeze_image_backbone,
        freeze_text_backbone=args.freeze_text_backbone,
        pretrained_image=pretrained_image,
    )

    return model