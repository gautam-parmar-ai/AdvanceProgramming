"""
Adaptive Knowledge Distillation

Computes:
1. KD Loss (KL Divergence)
2. Adaptive alpha and beta weights

Alpha and beta are adjusted dynamically using:
- Student prediction entropy
- Previous task accuracy (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveKD(nn.Module):

    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    # ----------------------------------
    # Prediction entropy
    # ----------------------------------
    def entropy(self, logits):

        probs = F.softmax(logits, dim=1)

        entropy = -torch.sum(
            probs * torch.log(probs + 1e-8),
            dim=1
        )

        return entropy.mean()

    # ----------------------------------
    # Adaptive weights
    # ----------------------------------
    def compute_weights(self, student_logits, prev_acc=None):

        ent = self.entropy(student_logits).item()

        # Higher entropy -> stronger KD
        alpha = 1.0 / (1.0 + ent)

        if prev_acc is not None:
            alpha *= (1.0 + (1.0 - prev_acc))

        # Feature distillation weight
        beta = alpha * 0.7

        return float(alpha), float(beta)

    # ----------------------------------
    # KL Distillation Loss
    # ----------------------------------
    def kd_loss(self, student_logits, teacher_logits):

        T = self.temperature

        student_probs = F.log_softmax(
            student_logits / T,
            dim=1
        )

        teacher_probs = F.softmax(
            teacher_logits / T,
            dim=1
        )

        loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction="batchmean"
        )

        return loss * (T * T)

    # ----------------------------------
    # Main forward call
    # ----------------------------------
    def forward(
        self,
        student_logits,
        teacher_logits,
        prev_acc=None
    ):

        kd_value = self.kd_loss(
            student_logits,
            teacher_logits
        )

        alpha, beta = self.compute_weights(
            student_logits,
            prev_acc
        )

        return kd_value, alpha, beta