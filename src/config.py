import os

# Data configuration
DATASET_ZIP_PATH = "/content/Dataset.zip"  # Update path as needed or get it from download_dataset.sh
EXTRACT_DIR = "/content"
SOURCE_DIR = os.path.join(EXTRACT_DIR, "Dataset")
ORGANIZED_DATA_DIR = os.path.join(EXTRACT_DIR, "dataset", "organized_classes")
IMAGES_PER_CLASS = 80

# Model and training configuration
NUM_CLASSES = 17
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
LEARNING_RATE_FREEZE = 0.001
LEARNING_RATE_FINETUNE = 0.0001

# Pre-trained model variant (EfficientNet-b3 used here)
MODEL_VARIANT = 'efficientnet-b3'
