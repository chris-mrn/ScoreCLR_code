from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from scoreclr.models import Encoder


class SSLMethod(nn.Module):
    """Structure shared by every method: the encoder (backbone + projector + online classifier,
    `self.encoder`) and an objective on the codes z0, z1 of the two views of a batch.

    A method implements `loss`. A method with an auxiliary network trained by its own optimizer
    (ScoreCLR's score net) also implements `auxiliary_parameters` and `auxiliary_step`, which the
    trainer runs on the detached codes before every encoder update. A method with extra trainable
    heads updated together with the encoder (BYOL's predictor) returns them from
    `encoder_extra_parameters`; one with a target network (BYOL's EMA encoder) implements
    `target_codes` and `after_step`."""

    def __init__(self, cfg, num_classes: int, small_images: bool):
        super().__init__()
        self.encoder = Encoder(cfg.model, num_classes, small_images)

    def loss(self, z0: torch.Tensor, z1: torch.Tensor, targets: Optional[List[torch.Tensor]] = None
             ) -> Dict[str, torch.Tensor]:
        """{"loss": the objective, with grad; other entries: detached terms, for logging}.
        `targets`: the output of `target_codes` (None for methods without a target network)."""
        raise NotImplementedError

    def encoder_extra_parameters(self) -> Iterable[nn.Parameter]:
        """Trainable parameters outside the encoder that share its optimizer, lr, weight decay and clipping."""
        return []

    def target_codes(self, views: List[torch.Tensor]) -> Optional[List[torch.Tensor]]:
        """Codes of each view from a target network, without grad (None: no target network)."""
        return None

    def after_step(self, step: int, total_steps: int) -> None:
        """Called after encoder update `step` (0-based) of `total_steps`."""

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        return []

    def auxiliary_step(self, z: torch.Tensor, optimizer: Optional[torch.optim.Optimizer]) -> Dict[str, torch.Tensor]:
        """Updates the auxiliary network on the detached codes z of both views; returns logged terms."""
        return {}

    def distribute(self, device_ids) -> None:
        """Wraps the auxiliary network for data-parallel training (the trainer wraps the encoder)."""
