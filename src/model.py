# ============================================================
# Model: ResNet18 + DistilBERT for MVSA 6-Class Classification
# ============================================================

import torch
import torch.nn as nn

from torchvision import models
from transformers import AutoModel


class StrongMultimodalNet(nn.Module):
    """
    Strong multimodal model:
        Image encoder: ResNet18
        Text encoder: DistilBERT
        Fusion: MLP projector
        Classifier: 6-class sentiment-strength classification
    """

    def __init__(
        self,
        text_model_name="distilbert-base-uncased",
        num_classes=6,
        image_hidden_dim=128,
        text_hidden_dim=128,
        projector_hidden_dim=128,
        dropout=0.1,
        freeze_image_backbone=True,
        freeze_text_backbone=True,
        pretrained_image=True,
    ):
        super().__init__()

        # -------------------------
        # Image encoder: ResNet18
        # -------------------------
        try:
            if pretrained_image:
                weights = models.ResNet18_Weights.DEFAULT
            else:
                weights = None

            resnet = models.resnet18(weights=weights)

        except Exception:
            # fallback for older torchvision versions
            resnet = models.resnet18(pretrained=pretrained_image)

        self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.image_proj = nn.Linear(512, image_hidden_dim)

        if freeze_image_backbone:
            for p in self.image_backbone.parameters():
                p.requires_grad = False

        # -------------------------
        # Text encoder: DistilBERT
        # -------------------------
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        bert_hidden = self.text_backbone.config.hidden_size
        self.text_proj = nn.Linear(bert_hidden, text_hidden_dim)

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
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(projector_hidden_dim, num_classes)

    def forward(self, image, input_ids, attention_mask):
        # Image branch
        image_feat = self.image_backbone(image)
        image_feat = image_feat.view(image_feat.size(0), -1)
        image_feat = self.image_proj(image_feat)

        # Text branch
        text_outputs = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # DistilBERT: use first token representation
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_feat)

        # Fusion
        fused = torch.cat([image_feat, text_feat], dim=1)
        h = self.multi_modal_projector(fused)
        h = self.dropout(h)

        logits = self.classifier(h)

        return logits


def build_model(args):
    """
    Build model from args/config.
    """

    model = StrongMultimodalNet(
        text_model_name=args.text_model_name,
        num_classes=args.num_classes,
        image_hidden_dim=args.image_hidden_dim,
        text_hidden_dim=args.text_hidden_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        dropout=args.dropout,
        freeze_image_backbone=args.freeze_image_backbone,
        freeze_text_backbone=args.freeze_text_backbone,
        pretrained_image=args.pretrained_image,
    )

    return model