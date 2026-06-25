

import numpy as np


class ForgettingMetrics:

    def __init__(self, num_tasks):

        self.num_tasks = num_tasks

        # Store accuracy matrix
        # rows = after training task t
        # cols = performance on task k
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))

    # -----------------------------
    # Log accuracy after each task
    # -----------------------------
    def update(self, task_id, accuracies):

        """
        accuracies: list of accuracies on all tasks seen so far
        """

        for i, acc in enumerate(accuracies):
            self.accuracy_matrix[task_id][i] = acc

    # -----------------------------
    # Average Accuracy
    # -----------------------------
    def average_accuracy(self):

        final_acc = self.accuracy_matrix[-1]

        return np.mean(final_acc[:self.num_tasks])

    # -----------------------------
    # Forgetting measure
    # -----------------------------
    def forgetting(self):

        forgetting = 0.0

        for task in range(self.num_tasks - 1):

            max_acc = np.max(self.accuracy_matrix[:, task])
            final_acc = self.accuracy_matrix[-1][task]

            forgetting += (max_acc - final_acc)

        return forgetting / (self.num_tasks - 1)

    # -----------------------------
    # Backward Transfer
    # -----------------------------
    def backward_transfer(self):

        bwt = 0.0

        for task in range(self.num_tasks - 1):

            bwt += self.accuracy_matrix[-1][task] - self.accuracy_matrix[task][task]

        return bwt / (self.num_tasks - 1)

    # -----------------------------
    # Retention score
    # -----------------------------
    def retention(self):

        final = self.accuracy_matrix[-1]

        return np.mean(final[:-1])