# Semi-Supervised Contrastive Learning for Audio Classification 🎵

This repository contains the implementation of a **Semi-Supervised Contrastive Learning** approach for classifying environmental sounds, tested on the **ESC-50 Dataset**.

The project demonstrates how a model can learn robust audio representations from unlabeled data (Pre-training) and achieve high accuracy with limited labeled data (Fine-tuning).

## 📊 Results (Animal Subset)

The model was fine-tuned on the **Animals Subset** (10 Classes) of ESC-50, achieving **93.75% Accuracy**.

![Confusion Matrix](matrix_hewan.png)

## 🚀 Features
- **Stage 1 (Pre-training):** Self-supervised learning using SimCLR-style contrastive loss (`NTXentLoss`) on unlabeled audio.
- **Stage 2 (Fine-tuning):** Supervised training on labeled data using the pre-trained encoder.
- **Domain Specificity:** Tested specifically on animal sounds (Dog, Cat, Crow, etc.) to prove domain adaptation.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/USERNAME/Semi-Supervised-Contrastive-Learning-Audio.git](https://github.com/USERNAME/Semi-Supervised-Contrastive-Learning-Audio.git)
   cd Semi-Supervised-Contrastive-Learning-Audio