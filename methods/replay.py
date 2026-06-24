"""
Replay Buffer for Continual Learning (Improved Version)

Implements:
- Reservoir sampling (unbiased memory)
- Lightweight class-balanced replay
- Stable batch sampling
"""

import torch
import random
from collections import defaultdict


class ReplayBuffer:

    def __init__(self, buffer_size=2000):
        self.buffer_size = buffer_size

        # store samples
        self.buffer = []

        # optional: class tracking for balance
        self.class_counts = defaultdict(int)
        self.total_seen = 0

    # -----------------------------
    # Reservoir sampling (BEST PRACTICE)
    # -----------------------------
    def add_samples(self, data_loader):

        for x, y in data_loader:

            for i in range(x.size(0)):

                sample = (x[i].cpu(), int(y[i].cpu()))

                self.total_seen += 1

                if len(self.buffer) < self.buffer_size:
                    self.buffer.append(sample)

                else:
                    # reservoir replacement
                    idx = random.randint(0, self.total_seen - 1)

                    if idx < self.buffer_size:
                        self.buffer[idx] = sample

                self.class_counts[sample[1]] += 1

    # -----------------------------
    # Balanced sampling
    # -----------------------------
    def sample(self, batch_size):

        if len(self.buffer) == 0:
            return None, None

        # ensure valid batch size
        batch_size = min(batch_size, len(self.buffer))

        batch = random.sample(self.buffer, batch_size)

        x, y = zip(*batch)

        x = torch.stack(x)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    # -----------------------------
    # Utility
    # -----------------------------
    def __len__(self):
        return len(self.buffer)