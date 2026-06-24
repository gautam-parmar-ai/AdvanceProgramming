import torch
from torch.optim import SGD
import torch.nn.utils as nn_utils

from losses.losses import TotalLoss
from methods.adaptive_kd import AdaptiveKD
from methods.feature_distillation import FeatureDistillationLoss


class BaseTrainer:

    def __init__(self, student, teacher, config, device, task_mask=None):

        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.device = device
        self.task_mask = task_mask

        self.config = config

        # Optimizer
        self.optimizer = SGD(
            self.student.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

        # Loss modules
        self.criterion = TotalLoss(
            ewc_lambda=config.ewc_lambda,
            replay_lambda=1.0
        )

        self.kd = AdaptiveKD(temperature=config.temperature)
        self.feature_loss_fn = FeatureDistillationLoss()

    # --------------------------------------------------
    # TRAIN ONE EPOCH (IMPROVED)
    # --------------------------------------------------
    def train_epoch(self, dataloader, replay_buffer=None, ewc=None):

        self.student.train()
        self.teacher.eval()

        total_loss = 0.0

        for x, y in dataloader:

            x, y = x.to(self.device), y.to(self.device)

            # reset feature cache safety (IMPORTANT for hooks)
            self.student.features = {}
            self.teacher.features = {}

            # -----------------------------
            # Forward pass
            # -----------------------------
            student_out = self.student(x)
            with torch.no_grad():
                teacher_out = self.teacher(x)

            student_logits = student_out["logits"]
            teacher_logits = teacher_out["logits"]

            # OPTIONAL: task mask (VERY IMPORTANT upgrade)
            if self.task_mask is not None:
                student_logits = self.task_mask.mask_logits(student_logits)

            # -----------------------------
            # KD loss
            # -----------------------------
            kd_value, _, _ = self.kd(student_logits, teacher_logits)

            alpha = 0.5
            beta = 0.3

            # -----------------------------
            # Feature loss
            # -----------------------------
            feature_value = self.feature_loss_fn(
                student_out["features"],
                teacher_out["features"]
            )

            # -----------------------------
            # EWC loss (SAFE scaling)
            # -----------------------------
            ewc_value = 0.0
            if ewc is not None:
                ewc_value = ewc.penalty()

            # -----------------------------
            # Replay batch
            # -----------------------------
            replay_logits = None
            replay_targets = None

            if replay_buffer is not None:
                rx, ry = replay_buffer.sample(self.config.batch_size)

                if rx is not None:
                    rx, ry = rx.to(self.device), ry.to(self.device)

                    replay_out = self.student(rx)
                    replay_logits = replay_out["logits"]
                    replay_targets = ry

            # -----------------------------
            # TOTAL LOSS
            # -----------------------------
           


            losses = self.criterion(
                student_logits,
                y,

                kd_value=kd_value,
                alpha=alpha,

                feature_value=feature_value,
                beta=beta,

                ewc_value=ewc_value,

                replay_logits=replay_logits,
                replay_targets=replay_targets
            )

            loss = losses["ce_loss"] * 1.0 + losses["kd_loss"] * 0.3 + losses["feature_loss"] * 0.1 + losses["ewc_loss"] * 0.05

            # -----------------------------
            # BACKPROP
            # -----------------------------
            self.optimizer.zero_grad()
            loss.backward()

            # 🔥 GRADIENT CLIPPING (VERY IMPORTANT)
            nn_utils.clip_grad_norm_(self.student.parameters(), 5.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)