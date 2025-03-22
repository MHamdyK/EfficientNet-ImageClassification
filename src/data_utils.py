# src/data_utils.py

import os
import zipfile
import shutil
from src.config import DATASET_ZIP_PATH, EXTRACT_DIR, SOURCE_DIR, ORGANIZED_DATA_DIR, IMAGES_PER_CLASS

def extract_dataset(zip_path=DATASET_ZIP_PATH, extract_dir=EXTRACT_DIR):
    """Extract the dataset zip file."""
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dataset extracted.")

def organize_dataset(source_dir=SOURCE_DIR, target_dir=ORGANIZED_DATA_DIR, images_per_class=IMAGES_PER_CLASS):
    """Organize images into class folders."""
    os.makedirs(target_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.jpg')])
    total_images = len(image_files)
    total_classes = total_images // images_per_class
    print(f"Found {total_classes} classes with a total of {total_images} images.")

    for idx, image_file in enumerate(image_files):
        class_number = (idx // images_per_class) + 1
        class_folder = os.path.join(target_dir, f"class_{class_number}")
        os.makedirs(class_folder, exist_ok=True)
        src_path = os.path.join(source_dir, image_file)
        dest_path = os.path.join(class_folder, image_file)
        print(f"Moving: {src_path} -> {dest_path}")
        shutil.move(src_path, dest_path)
    print("Dataset organized.")

def remove_checkpoints(target_dir=ORGANIZED_DATA_DIR):
    """Remove all .ipynb_checkpoints folders recursively."""
    checkpoints_path = os.path.join(target_dir, ".ipynb_checkpoints")
    if os.path.exists(checkpoints_path):
        shutil.rmtree(checkpoints_path)
        print("Removed '.ipynb_checkpoints' folder.")
    else:
        print("No '.ipynb_checkpoints' folder found.")
