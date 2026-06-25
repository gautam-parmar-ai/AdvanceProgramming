

import torch
import numpy as np
from torch.utils.data import DataLoader

class Evaluator:

    def __init__(self):
        self.accuracy_matrix = []

    # -----------------------------
    # Evaluate model on one task
    # -----------------------------
    def evaluate_task(self, model, dataset, device, task_classes=None):

        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False
        )

        correct = 0
        total = 0

        # Optional: restrict predictions to the task's class subset
        # (recommended for class-incremental/task-level accuracy)
        with torch.no_grad():

            for x, y in loader:

                x = x.to(device)
                y = y.to(device)

                outputs = model(x)["logits"]

                if task_classes is not None:
                    task_classes_tensor = torch.tensor(task_classes, device=outputs.device, dtype=torch.long)
                    masked = outputs.clone()
                    masked[:, task_classes_tensor] = masked[:, task_classes_tensor]
                    masked[:, [c for c in range(outputs.size(1)) if c not in set(task_classes)]] = float("-inf")
                    preds = masked.argmax(dim=1)
                else:
                    preds = outputs.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total

    # -----------------------------
    # Update accuracy matrix
    # -----------------------------
    def update(self, model, tasks, task_id, device):

        row = []

        for j in range(task_id + 1):

            acc = self.evaluate_task(
                model,
                tasks[j]["test"],
                device,
                task_classes=tasks[j]["classes"]
            )

            row.append(acc)

        self.accuracy_matrix.append(row)

    # -----------------------------
    # Compute metrics
    # -----------------------------
    def compute_metrics(self):

        A = np.array(self.accuracy_matrix, dtype=object)

        num_tasks = len(A)

        # Final accuracy
        final_acc = np.mean(A[-1])

        # Average accuracy over all tasks
        avg_acc = np.mean([np.mean(row) for row in A])

        # Forgetting
        forgetting = 0.0

        for j in range(num_tasks):

            max_acc = 0.0
            final_task_seen = A[-1][j]

            for i in range(j, num_tasks):
                if j < len(A[i]):
                    max_acc = max(max_acc, A[i][j])

            forgetting += (max_acc - final_task_seen)

        forgetting /= num_tasks

        # Backward Transfer
        bwt = 0.0

        for j in range(num_tasks):

            if len(A) > 1 and j < len(A[-1]):
                bwt += (A[-1][j] - A[j][j])

        bwt /= num_tasks

        return {
            "final_accuracy": final_acc,
            "average_accuracy": avg_acc,
            "forgetting": forgetting,
            "backward_transfer": bwt,
            "accuracy_matrix": self.accuracy_matrix
        }