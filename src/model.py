# ============================================================
# Model: CLIP Dual-Encoder with Missing-Modality Fusion
# For Hateful Memes 2-class classification
#
# Supports:
#   image_only
#   text_only
#   modality_exclusive
#   full_multimodal
#
# Encoders:
#   Image encoder: CLIP image encoder
#   Text encoder : CLIP text encoder
#
# Fusion:
#   image_feat
#   text_feat
#   image_feat * text_feat
#   |image_feat - text_feat|
# ============================================================

import torch
import torch.nn as nn

from transformers import CLIPModel


class StrongMultimodalNet(nn.Module):
    """
    CLIP dual-encoder model with missing-modality fusion.

    This model is designed for heterogeneous multimodal FL:
        image_only:
            use CLIP image encoder, replace text with learnable missing embedding
        text_only:
            use CLIP text encoder, replace image with learnable missing embedding
        modality_exclusive:
            image clients use image branch
            text clients use text branch
        full_multimodal:
            use both image and text
    """

    def __init__(
        self,
        text_model_name="openai/clip-vit-base-patch32",
        image_model_name="openai/clip-vit-base-patch32",
        num_classes=2,
        image_hidden_dim=256,
        text_hidden_dim=256,
        projector_hidden_dim=256,
        dropout=0.5,
        freeze_image_backbone=False,
        freeze_text_backbone=False,
        pretrained_image=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.projector_hidden_dim = projector_hidden_dim

        # ------------------------------------------------------------
        # CLIP model
        # ------------------------------------------------------------
        self.clip = CLIPModel.from_pretrained(image_model_name)

        clip_proj_dim = self.clip.config.projection_dim

        # ------------------------------------------------------------
        # Image projection
        # ------------------------------------------------------------
        self.image_proj = nn.Sequential(
            nn.Linear(clip_proj_dim, image_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(image_hidden_dim, projector_hidden_dim),
        )

        # ------------------------------------------------------------
        # Text projection
        # ------------------------------------------------------------
        self.text_proj = nn.Sequential(
            nn.Linear(clip_proj_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_hidden_dim, projector_hidden_dim),
        )

        # ------------------------------------------------------------
        # Freeze CLIP branches if needed
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------
        self.image_norm = nn.LayerNorm(projector_hidden_dim)
        self.text_norm = nn.LayerNorm(projector_hidden_dim)

        # ------------------------------------------------------------
        # Learnable missing-modality embeddings
        # ------------------------------------------------------------
        self.missing_image_embedding = nn.Parameter(
            torch.zeros(1, projector_hidden_dim)
        )
        self.missing_text_embedding = nn.Parameter(
            torch.zeros(1, projector_hidden_dim)
        )

        nn.init.normal_(self.missing_image_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_text_embedding, mean=0.0, std=0.02)

        # ------------------------------------------------------------
        # Fusion projector
        # ------------------------------------------------------------
        # Fusion vector:
        #   [image_feat, text_feat, image_feat * text_feat, |image_feat - text_feat|]
        #
        # Keep the name "multi_modal_projector" so update extraction still works.
        self.multi_modal_projector = nn.Sequential(
            nn.Linear(projector_hidden_dim * 4, projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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
        Compatible with existing training code:
            model(
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                setting=mode,
            )
        """

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

        # ------------------------------------------------------------
        # Image branch
        # ------------------------------------------------------------
        if use_image and pixel_values is not None:
            image_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
            )

            image_pooled = image_outputs.pooler_output
            image_feat = self.clip.visual_projection(image_pooled)

            image_feat = self.image_proj(image_feat)
            image_feat = self.image_norm(image_feat)
        else:
            image_feat = self.missing_image_embedding.expand(
                batch_size,
                -1,
            ).to(device)

        # ------------------------------------------------------------
        # Text branch
        # ------------------------------------------------------------
        if use_text and input_ids is not None and attention_mask is not None:
            text_outputs = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            text_pooled = text_outputs.pooler_output
            text_feat = self.clip.text_projection(text_pooled)

            text_feat = self.text_proj(text_feat)
            text_feat = self.text_norm(text_feat)
        else:
            text_feat = self.missing_text_embedding.expand(
                batch_size,
                -1,
            ).to(device)

        # ------------------------------------------------------------
        # Fusion
        # ------------------------------------------------------------
        interaction = image_feat * text_feat
        difference = torch.abs(image_feat - text_feat)

        fused = torch.cat(
            [
                image_feat,
                text_feat,
                interaction,
                difference,
            ],
            dim=1,
        )

        h = self.multi_modal_projector(fused)
        logits = self.classifier(h)

        return logits


def build_model(args):
    """
    Build CLIP dual-encoder Hateful Memes model from args/config.
    """

    image_model_name = getattr(
        args,
        "image_model_name",
        "openai/clip-vit-base-patch32",
    )

    text_model_name = getattr(
        args,
        "text_model_name",
        "openai/clip-vit-base-patch32",
    )

    pretrained_image = getattr(args, "pretrained_image", True)

    model = StrongMultimodalNet(
        text_model_name=text_model_name,
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