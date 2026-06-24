from dataclasses import dataclass

@dataclass
class Config:

    # Dataset
    dataset_name = "CIFAR100"
    num_classes = 100
    num_tasks = 10
    classes_per_task = 10

    # Models
    teacher_model = "resnet50"
    student_model = "resnet18"

    # Training
    batch_size = 128
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-4
    momentum = 0.9

    # Distillation
    temperature = 4.0

    # Adaptive KD
    alpha_min = 0.1
    alpha_max = 0.5

    beta_min = 0.1
    beta_max = 0.3

    # EWC
    ewc_lambda = 50

    # Replay
    replay_buffer_size = 2000

    # Logging
    save_dir = "checkpoints"
    log_dir = "logs"
    result_dir = "results"

    # Reproducibility
    seed = 42

    # Device
    device = "cuda"