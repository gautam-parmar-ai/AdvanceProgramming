

import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalLoss(nn.Module):

    def __init__(self, ewc_lambda=5000, replay_lambda=1.0):

        super(TotalLoss, self).__init__()

        self.ewc_lambda = ewc_lambda
        self.replay_lambda = replay_lambda

        self.ce_loss = nn.CrossEntropyLoss()

    # -----------------------------
    # Cross entropy loss
    # -----------------------------
    def classification_loss(self, student_logits, targets):

        return self.ce_loss(student_logits, targets)

    # -----------------------------
    # KD loss already computed externally
    # -----------------------------
    def kd_loss(self, kd_value, alpha):

        return alpha * kd_value

    # -----------------------------
    # Feature loss
    # -----------------------------
    def feature_loss(self, feature_loss_value, beta):

        return beta * feature_loss_value

    # -----------------------------
    # Replay loss
    # -----------------------------
    def replay_loss(self, student_logits, replay_logits, replay_targets):

        if replay_logits is None:
            return 0.0

        return self.replay_lambda * F.cross_entropy(replay_logits, replay_targets)

    # -----------------------------
    # Full loss
    # -----------------------------
    def forward(
        self,
        student_logits,
        targets,

        kd_value=0.0,
        alpha=1.0,

        feature_value=0.0,
        beta=1.0,

        ewc_value=0.0,

        replay_logits=None,
        replay_targets=None
    ):

        ce = self.classification_loss(student_logits, targets)

        kd = self.kd_loss(kd_value, alpha)

        feat = self.feature_loss(feature_value, beta)

        ewc = self.ewc_lambda * ewc_value

        replay = self.replay_loss(student_logits, replay_logits, replay_targets)

        total = ce + 2.0 * kd + 1.5 * feat + ewc + replay

        return {
            "total_loss": total,
            "ce_loss": ce,
            "kd_loss": kd,
            "feature_loss": feat,
            "ewc_loss": ewc,
            "replay_loss": replay
        }