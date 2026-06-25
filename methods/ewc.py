

import torch
import torch.nn.functional as F


class EWC:

    def __init__(self, model, dataloader, device):

        self.model = model
        self.device = device

        # Store parameter snapshot after training on task
        self.params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # Fisher Information Matrix
        self.fisher = self._compute_fisher(dataloader)

    # -----------------------------
    # Fisher computation
    # -----------------------------
    def _compute_fisher(self, dataloader):

        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.params.items()
        }

        self.model.eval()

        for x, y in dataloader:

            x, y = x.to(self.device), y.to(self.device)

            self.model.zero_grad()

            output = self.model(x)

            loss = F.cross_entropy(output["logits"], y)
            loss.backward()

            for n, p in self.model.named_parameters():

                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2

        # Normalize
        for n in fisher:
            fisher[n] /= len(dataloader)

        return fisher

    # -----------------------------
    # EWC penalty loss
    # -----------------------------
    def penalty(self):

        loss = 0.0

        for n, p in self.model.named_parameters():

            if n in self.fisher:

                loss += torch.sum(
                    self.fisher[n] * (p - self.params[n]) ** 2
                )

        return loss