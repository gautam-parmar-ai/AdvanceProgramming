

import torch


class TaskMask:

    def __init__(self, classes_per_task):
        self.classes_per_task = classes_per_task
        self.seen_classes = []

    def update(self, task_id):

        start = task_id * self.classes_per_task
        end = start + self.classes_per_task

        self.seen_classes = list(range(end))

    def mask_logits(self, logits):

        mask = torch.ones_like(logits) * float("-inf")
        mask[:, self.seen_classes] = 0

        return logits + mask