from typing import List, Tuple

import torch
import torch.nn as nn

from scoreclr.models.backbone import resnet


class Encoder(nn.Module):
    """Backbone + MLP projector (Linear-ReLU-Linear, no BatchNorm), shared by every method, and an
    online linear classifier on the detached backbone features (monitoring only)."""

    def __init__(self, model_cfg, num_classes: int, small_images: bool):
        super().__init__()
        self.backbone = resnet(model_cfg.backbone, small_images)
        feat_dim = self.backbone.out_dim
        if model_cfg.get("projector", True):
            self.projector = nn.Sequential(
                nn.Linear(feat_dim, model_cfg.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_cfg.proj_hidden_dim, model_cfg.proj_dim),
            )
            self.code_dim = model_cfg.proj_dim
        else:  # model.projector=false: the codes are the backbone features
            self.projector, self.code_dim = nn.Identity(), feat_dim
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, views: List[torch.Tensor]) -> Tuple[List[torch.Tensor], ...]:
        """Each view is forwarded separately (per-view BatchNorm statistics).
        Returns lists of (backbone features, codes z, classifier logits)."""
        feats = [self.backbone(x) for x in views]
        return feats, [self.projector(f) for f in feats], [self.classifier(f.detach()) for f in feats]

    def param_groups(self, lr: float, weight_decay: float, classifier_lr: float) -> List[dict]:
        encoder = list(self.backbone.parameters()) + list(self.projector.parameters())
        return [
            {"params": encoder, "lr": lr, "weight_decay": weight_decay},
            {"params": self.classifier.parameters(), "lr": classifier_lr, "weight_decay": 0.0},
        ]

