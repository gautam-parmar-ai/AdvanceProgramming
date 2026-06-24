"""
Main training script for Continual Learning project.

Runs:
- Task incremental learning
- Baselines + proposed method
- Correct continual learning evaluation
- Full visualizations (research-ready)
"""

import torch

from configs.config import Config
from datasets.cifar100 import CIFAR100DataModule
from datasets.task_split import TaskSplitter

from models.teacher import TeacherNet
from models.student import StudentNet

from trainers.base_trainer import BaseTrainer

from methods.replay import ReplayBuffer
from methods.ewc import EWC

# Evaluation
from metrics.evaluator import Evaluator

# Visualization (NEW)
from visualization.plots import CLVisualizer


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load dataset
    # -----------------------------
    data_module = CIFAR100DataModule()
    train_dataset, test_dataset = data_module.get_datasets()

    splitter = TaskSplitter(train_dataset, test_dataset)
    tasks = splitter.get_all_tasks()

    # -----------------------------
    # Models
    # -----------------------------
    teacher = TeacherNet(num_classes=Config.num_classes)

    teacher.load_state_dict(torch.load("teacher_cifar100.pth", map_location=device))

    teacher = teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False
    student = StudentNet(num_classes=Config.num_classes)

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = BaseTrainer(student, teacher, Config, device)

    # -----------------------------
    # Replay + EWC
    # -----------------------------
    replay = ReplayBuffer(Config.replay_buffer_size)
    ewc = None

    # -----------------------------
    # Evaluator
    # -----------------------------
    evaluator = Evaluator()

    # -----------------------------
    # Visualizer (NEW)
    # -----------------------------
    viz = CLVisualizer()

    # -----------------------------
    # Task loop
    # -----------------------------
    for task_id, task in enumerate(tasks):

        print(f"\n===== Training Task {task_id} =====")
        

        train_loader = torch.utils.data.DataLoader(
            task["train"],
            batch_size=Config.batch_size,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            task["test"],
            batch_size=Config.batch_size,
            shuffle=False
        )

        # -----------------------------
        # Train
        # -----------------------------
        for epoch in range(Config.epochs):

            loss = trainer.train_epoch(
                train_loader,
                replay_buffer=replay,
                ewc=ewc
            )

            print(f"Task {task_id} Epoch {epoch} Loss: {loss:.4f}")

        # -----------------------------
        # Update EWC
        # -----------------------------
        ewc = EWC(student, train_loader, device)

        # -----------------------------
        # Update Replay Buffer
        # -----------------------------
        replay.add_samples(train_loader)

        # -----------------------------
        # Correct Evaluation
        # -----------------------------
        evaluator.update(
            student,
            tasks,
            task_id,
            device
        )

    # -----------------------------
    # FINAL METRICS
    # -----------------------------
    results = evaluator.compute_metrics()

    print("\n===== FINAL RESULTS =====")

    print("Final Accuracy:", results["final_accuracy"])
    print("Average Accuracy:", results["average_accuracy"])
    print("Forgetting:", results["forgetting"])
    print("Backward Transfer:", results["backward_transfer"])

    # -----------------------------
    # VISUALIZATIONS (NEW)
    # -----------------------------
    print("\n===== GENERATING VISUALIZATIONS =====")

    accuracy_matrix = results["accuracy_matrix"]

    viz.plot_accuracy(accuracy_matrix)
    viz.plot_forgetting(accuracy_matrix)
    viz.plot_backward_transfer(accuracy_matrix)
    viz.plot_heatmap(accuracy_matrix)


if __name__ == "__main__":
    main()