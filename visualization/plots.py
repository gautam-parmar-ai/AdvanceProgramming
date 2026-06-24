import numpy as np
import matplotlib.pyplot as plt


class CLVisualizer:

    def __init__(self):
        pass

    # -----------------------------
    # 1. Accuracy over tasks
    # -----------------------------
    def plot_accuracy(self, accuracy_matrix):

        avg_acc = [np.mean(row) for row in accuracy_matrix]

        plt.figure()
        plt.plot(avg_acc, marker='o')
        plt.title("Average Accuracy Over Tasks")
        plt.xlabel("Task")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()

    # -----------------------------
    # 2. Forgetting curve
    # -----------------------------
    def plot_forgetting(self, accuracy_matrix):

        num_tasks = len(accuracy_matrix)

        forgetting = []

        for j in range(num_tasks):

            max_acc = 0
            final_acc = accuracy_matrix[-1][j]

            for i in range(len(accuracy_matrix)):
                if j < len(accuracy_matrix[i]):
                    max_acc = max(max_acc, accuracy_matrix[i][j])

            forgetting.append(max_acc - final_acc)

        plt.figure()
        plt.plot(forgetting, marker='o', color='red')
        plt.title("Forgetting per Task")
        plt.xlabel("Task")
        plt.ylabel("Forgetting")
        plt.grid()
        plt.show()

    # -----------------------------
    # 3. Backward Transfer
    # -----------------------------
    def plot_backward_transfer(self, accuracy_matrix):

        bwt = []

        for j in range(len(accuracy_matrix)):

            if j < len(accuracy_matrix[-1]):
                bwt.append(
                    accuracy_matrix[-1][j] - accuracy_matrix[j][j]
                )

        plt.figure()
        plt.plot(bwt, marker='o', color='green')
        plt.title("Backward Transfer per Task")
        plt.xlabel("Task")
        plt.ylabel("BWT")
        plt.grid()
        plt.show()

    # -----------------------------
    # 4. Accuracy Heatmap (VERY IMPORTANT)
    # -----------------------------
    def plot_heatmap(self, accuracy_matrix):

        mat = np.zeros((len(accuracy_matrix), len(accuracy_matrix)))

        for i in range(len(accuracy_matrix)):
            for j in range(len(accuracy_matrix[i])):
                mat[i][j] = accuracy_matrix[i][j]

        plt.figure()
        plt.imshow(mat, cmap='viridis')
        plt.colorbar()
        plt.title("Accuracy Matrix Heatmap (Task i → Task j)")
        plt.xlabel("Test Task")
        plt.ylabel("Train Task")
        plt.show()