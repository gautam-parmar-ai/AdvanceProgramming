import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.proj3 = None
        self.proj4 = None

    def _project(self, s, t, layer):

        if layer == "3":
            if self.proj3 is None:
                self.proj3 = nn.Conv2d(s.shape[1], t.shape[1], 1).to(s.device)
            s = self.proj3(s)

        if layer == "4":
            if self.proj4 is None:
                self.proj4 = nn.Conv2d(s.shape[1], t.shape[1], 1).to(s.device)
            s = self.proj4(s)

        return s, t

    def forward(self, student_features, teacher_features):

        loss = 0.0

        for layer in ["layer3", "layer4"]:

            s = student_features[layer]
            t = teacher_features[layer]

            if s is None or t is None:
                continue

            s, t = self._project(s, t, "3" if layer == "layer3" else "4")

            s = F.normalize(s.flatten(1), dim=1)
            t = F.normalize(t.flatten(1), dim=1)

            loss += 1 - F.cosine_similarity(s, t).mean()

        return loss