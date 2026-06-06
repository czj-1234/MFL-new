# ============================================================
# Model: CLIP-ViT + RoBERTa with Modality-Aware Gated Fusion
# For Hateful Memes 2-class classification
# ============================================================

import torch
import torch.nn as nn

from transformers import AutoModel, CLIPVisionModel


class StrongMultimodalNet(nn.Module):
    """
    Strong multimodal model with modality-aware gated fusion.

    Encoders:
        Image encoder: CLIP-ViT
        Text encoder: RoBERTa / BERT

    Fusion:
        1. Project image/text features into a shared hidden space.
        2. Use learnable missing-modality embeddings instead of zero vectors.
        3. Use a gate to learn image/text contribution.
        4. Classify from the gated fused representation.

    Suitable for modality_exclusive FL: each client may only have image or text.
    """

    def __init__(
        self,
        text_model_name="roberta-base",
        image_model_name="openai/clip-vit-base-patch32",
        num_classes=2,
        image_hidden_dim=256,
        text_hidden_dim=256,
        projector_hidden_dim=256,
        dropout=0.3,
        freeze_image_backbone=True,
        freeze_text_backbone=True,
        pretrained_image=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.projector_hidden_dim = projector_hidden_dim

        # -------------------------
        # Image encoder: CLIP-ViT
        # -------------------------
        self.image_backbone = CLIPVisionModel.from_pretrained(image_model_name)
        clip_hidden = self.image_backbone.config.hidden_size

        self.image_proj = nn.Sequential(
            nn.Linear(clip_hidden, image_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(image_hidden_dim, projector_hidden_dim),
        )

        if freeze_image_backbone:
            for p in self.image_backbone.parameters():
                p.requires_grad = False

        # -------------------------
        # Text encoder: RoBERTa / BERT
        # -------------------------
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        text_backbone_hidden = self.text_backbone.config.hidden_size

        self.text_proj = nn.Sequential(
            nn.Linear(text_backbone_hidden, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_hidden_dim, projector_hidden_dim),
        )

        if freeze_text_backbone:
            for p in self.text_backbone.parameters():
                p.requires_grad = False

        # -------------------------
        # Normalization
        # -------------------------
        self.image_norm = nn.LayerNorm(projector_hidden_dim)
        self.text_norm = nn.LayerNorm(projector_hidden_dim)

        # -------------------------
        # Learnable missing-modality embeddings
        # -------------------------
        self.missing_image_embedding = nn.Parameter(
            torch.zeros(1, projector_hidden_dim)
        )
        self.missing_text_embedding = nn.Parameter(
            torch.zeros(1, projector_hidden_dim)
        )

        nn.init.normal_(self.missing_image_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_text_embedding, mean=0.0, std=0.02)

        # -------------------------
        # Modality-aware gate
        # -------------------------
        self.fusion_gate = nn.Sequential(
            nn.Linear(projector_hidden_dim * 2, projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden_dim, 2),
            nn.Softmax(dim=1),
        )

        self.multi_modal_projector = nn.Sequential(
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
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

        use_image = setting in ["image", "image_only", "both", "multimodal", "modality_exclusive"]
        use_text = setting in ["text", "text_only", "both", "multimodal", "modality_exclusive"]

        # Image branch
        if use_image and pixel_values is not None:
            image_outputs = self.image_backbone(pixel_values=pixel_values)
            image_cls = image_outputs.pooler_output
            image_feat = self.image_proj(image_cls)
            image_feat = self.image_norm(image_feat)
        else:
            image_feat = self.missing_image_embedding.expand(batch_size, -1).to(device)

        # Text branch
        if use_text and input_ids is not None and attention_mask is not None:
            text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
            text_cls = text_outputs.last_hidden_state[:, 0, :]
            text_feat = self.text_proj(text_cls)
            text_feat = self.text_norm(text_feat)
        else:
            text_feat = self.missing_text_embedding.expand(batch_size, -1).to(device)

        # Gated fusion
        gate_input = torch.cat([image_feat, text_feat], dim=1)
        gate = self.fusion_gate(gate_input)
        image_weight = gate[:, 0:1]
        text_weight = gate[:, 1:2]
        fused = image_weight * image_feat + text_weight * text_feat

        h = self.multi_modal_projector(fused)
        h = self.dropout(h)

        logits = self.classifier(h)
        return logits


def build_model(args):
    """
    Build Hateful Memes 2-class model from args/config.
    """
    image_model_name = getattr(args, "image_model_name", "openai/clip-vit-base-patch32")
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