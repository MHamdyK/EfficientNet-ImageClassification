# EfficientNet-ImageClassification
# RPS-EfficientNet-TransferLearning

This repository demonstrates transfer learning using EfficientNet on an image classification dataset. The dataset consists of 17 classes (80 images per class) and is organized automatically. Two transfer learning approaches are implemented:
- **Frozen Base Model:** Only the classifier is trained.
- **Fine-Tuned Model:** The entire network is fine-tuned.

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [License](#license)

## Overview

This project shows how to:
- Download and organize a dataset (download link:(https://drive.google.com/file/d/1MNgjbLi7mB1wmX9SwL3JOdjBZjjM7VSi/view?usp=sharing)).
- Split the dataset into 80% train, 10% development, and 10% test sets.
- Apply two transfer learning strategies with EfficientNet (frozen base vs. full fine-tuning).
- Train for 50 epochs with early stopping on the development set.
- Plot training/validation accuracy and loss curves, classification reports, and confusion matrices.

## Directory Structure
```plaintext
EfficientNet-ImageClassification/
├── data/
│   ├── README.md             # Dataset download instructions
│   └── download_dataset.sh   # (Optional) Script to download and extract the dataset
├── notebooks/
│   └── efficientnet_transfer_learning.ipynb   # Colab/Jupyter notebook
├── src/
│   ├── __init__.py           # Package initializer
│   ├── config.py             # Configuration settings
│   ├── data_utils.py         # Dataset extraction and organization functions
│   ├── train.py              # Training routines for both transfer learning approaches
│   ├── evaluate.py           # Evaluation and plotting functions
│   └── utils.py              # Utility functions
├── requirements.txt          # List of required Python packages
├── README.md                 # Project overview and instructions
└── .gitignore                # Files/folders to ignore in Git
```

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/RPS-EfficientNet-TransferLearning.git
   cd RPS-EfficientNet-TransferLearning
2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

**Usage**
Two training approaches are provided, one with a Frozen backbone, and the other with the backbone unfrozen
