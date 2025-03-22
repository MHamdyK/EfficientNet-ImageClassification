# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from efficientnet_pytorch import EfficientNet
from src.config import NUM_CLASSES, EPOCHS, PATIENCE, LEARNING_RATE_FREEZE, LEARNING_RATE_FINETUNE

def create_efficientnet_b3(num_classes=NUM_CLASSES, freeze_base=True):
    """Create an EfficientNet-b3 model with a custom classifier."""
    model = EfficientNet.from_pretrained('efficientnet-b3')
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    return model

def train_model(model, train_loader, dev_loader, criterion, optimizer, epochs=EPOCHS, patience=PATIENCE, device=None, best_model_path='best_model.pt'):
    best_loss = np.inf
    patience_counter = 0
    train_losses, dev_losses = [], []
    train_acc, dev_acc = [], []
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train, correct_train, epoch_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        train_losses.append(epoch_loss / len(train_loader))
        train_acc.append(correct_train / total_train)
        
        # Validation phase
        model.eval()
        total_dev, correct_dev, dev_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in dev_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                dev_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_dev += (preds == labels).sum().item()
                total_dev += labels.size(0)
        dev_losses.append(dev_loss / len(dev_loader))
        dev_acc.append(correct_dev / total_dev)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Dev Loss: {dev_losses[-1]:.4f} | Train Acc: {train_acc[-1]:.4f} | Dev Acc: {dev_acc[-1]:.4f}")
        
        # Early stopping
        if dev_losses[-1] < best_loss:
            best_loss = dev_losses[-1]
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    return train_losses, dev_losses, train_acc, dev_acc
