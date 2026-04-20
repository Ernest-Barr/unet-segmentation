import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from dataset import get_dataset
from models import get_model


def setup_output_directory(model_name, dataset_name):
    weights_dir = os.path.join("..", "weights", model_name, dataset_name)
    plots_dir = os.path.join("..", "plots", model_name, dataset_name)

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return weights_dir, plots_dir


def plot(train_loss, val_loss, path, model_name, dataset_name, best_epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')

    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, val_loss[best_epoch - 1], color='red')

    plt.title(f'Loss Curves: {model_name} on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"{model_name}_{dataset_name}_loss.png"))
    plt.close()


def train_epoch(model, loader, optimizer, criterion, device):
    model = model.train()

    total_loss = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            loss = criterion(output, masks)
            total_loss += loss.item()

        return total_loss / len(loader)


# https://www.geeksforgeeks.org/deep-learning/understanding-pytorch-learning-rate-scheduling/
# https://d2l.ai/chapter_optimization/lr-scheduler.html
def train_models(model_name, dataset_name):
    weights_dir, plots_dir = setup_output_directory(model_name, dataset_name)

    train_ds = get_dataset(dataset_name, split="train", download=False)
    val_ds = get_dataset(dataset_name, split="val", download=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    model = get_model(model_name, imgChannels=3, outChannels=1).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_loss = float("inf")
    best_epoch_idx = 1
    patience_counter = 0

    train_history, val_history = [], []

    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, config.DEVICE)

        train_history.append(train_loss)
        val_history.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_{epoch}.pth"))

        if val_loss < best_val_loss - config.DELTA_PATIENCE:
            best_val_loss = val_loss
            best_epoch_idx = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_best.pth"))
        else:
            patience_counter += 1

        if patience_counter == config.PATIENCE:
            print(f"Early stopping, model has not improved for {config.PATIENCE} epochs")
            break

    plot(train_history, val_history, plots_dir, model_name, dataset_name, best_epoch_idx)


if __name__ == '__main__':
    for dataset in config.DATASETS:
        for model in config.MODELS:
            try:
                train_models(model, dataset)
            except Exception as e:
                print(e)
