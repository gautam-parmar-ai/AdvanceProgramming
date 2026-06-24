import torch
import torch.nn as nn
from torch.optim import SGD

from models.teacher import TeacherNet
from datasets.cifar100 import CIFAR100DataModule

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Data
# -----------------------
data_module = CIFAR100DataModule()
train_dataset, test_dataset = data_module.get_datasets()

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# -----------------------
# Model
# -----------------------
teacher = TeacherNet(num_classes=100).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = SGD(
    teacher.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)

# -----------------------
# Training
# -----------------------
epochs = 20

teacher.train()

for epoch in range(epochs):

    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = teacher(x)["logits"]

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Teacher Epoch {epoch}: Loss {total_loss/len(train_loader):.4f}")

# -----------------------
# Save model
# -----------------------
torch.save(teacher.state_dict(), "teacher_cifar100.pth")
print("✅ Teacher saved successfully!")