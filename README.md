# Adaptive Multi-View Knowledge Distillation for Continual Learning

## 📌 Project Overview

This project implements a **continual learning framework** for image classification using:

- Adaptive Multi-View Knowledge Distillation (AMVKD)
- Elastic Weight Consolidation (EWC)
- Experience Replay
- Feature-level distillation

The system is designed to mitigate **catastrophic forgetting** in class-incremental learning settings.

---

## 🧠 Key Idea

A student model learns tasks sequentially while:

- learning new classes
- retaining old knowledge
- adapting distillation weights dynamically

---

## 📊 Dataset

We use:

CIFAR-100 dataset

Split into:

- 10 tasks
- 10 classes per task

---

## 🏗️ Model Architecture

### Teacher

- ResNet50 (ImageNet pretrained)
- Frozen during training

### Student

- ResNet18
- Trained incrementally

---

## 🔬 Methods

- Knowledge Distillation (logit-level)
- Feature Distillation (layer3, layer4)
- Adaptive weighting (α(t), β(t))
- EWC regularization
- Replay buffer

---

## 📈 Evaluation Metrics

- Average Accuracy
- Forgetting Rate
- Backward Transfer (BWT)
- Retention Score

---

## 📁 Project Structure

See folder layout in repository.

---

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```
