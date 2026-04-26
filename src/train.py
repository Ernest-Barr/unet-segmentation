import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import config

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import get_dataset
from models import get_model

# https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
# https://medium.com/@devanshipratiher/understanding-loss-functions-for-deep-learning-segmentation-models-30187836b30a
# https://stackoverflow.com/questions/65125670/implementing-multiclass-dice-loss-function

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, task='binary'):
        if task == 'binary':
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)
            targets = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        probs = probs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, task='binary'):
        super(CombinedLoss, self).__init__()
        self.task = task
        self.dice = DiceLoss()
        if task == 'binary':
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets, task=self.task)
        return ce_loss + dice_loss

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


def train_epoch(model, model_name, dataset_name, loader, optimizer, criterion, device):
    model = model.train()

    total_loss = 0

    progress_bar = tqdm(loader, desc=f"Training {model_name} on {dataset_name} ", leave=False)

    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)

        if criterion.task == 'multiclass':
            masks = masks.squeeze(1).long()

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def validate_epoch(model, model_name, dataset_name, loader, criterion, device):
    model.eval()
    total_loss = 0

    progress_bar = tqdm(loader, desc=f"Validating {model_name} on {dataset_name} ", leave=False)

    with torch.no_grad():
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            if criterion.task == 'multiclass':
                masks = masks.squeeze(1).long()

            output = model(images)
            loss = criterion(output, masks)
            total_loss += loss.item()

        return total_loss / len(loader)

# https://www.geeksforgeeks.org/deep-learning/understanding-pytorch-learning-rate-scheduling/
# https://d2l.ai/chapter_optimization/lr-scheduler.html
def train_model(model_name, dataset_name):
    print(f"\n{'='*80}")
    print(f"Training: {model_name} on {dataset_name}")
    print(f"{'='*80}\n")
    
    weights_dir, plots_dir = setup_output_directory(model_name, dataset_name)

    train_ds, metadata = get_dataset(dataset_name, split="train", download=False)
    val_ds, _ = get_dataset(dataset_name, split="val", download=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    model = get_model(model_name, imgChannels=3, outChannels=metadata['out_channels']).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE)


    criterion = CombinedLoss(task=metadata['task'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_loss = float("inf")
    best_epoch_idx = 1
    patience_counter = 0

    train_history, val_history = [], []

    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = train_epoch(model, model_name, dataset_name, train_loader, optimizer, criterion, config.DEVICE)
        val_loss = validate_epoch(model, model_name, dataset_name, val_loader, criterion, config.DEVICE)

        train_history.append(train_loss)
        val_history.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{model_name} | {dataset_name}] Epoch {epoch:03d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Patience: {patience_counter} / {config.PATIENCE}")

        torch.save(model.state_dict(), os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_{epoch}.pth"))

        if val_loss < best_val_loss - config.DELTA_PATIENCE:
            best_val_loss = val_loss
            best_epoch_idx = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_best.pth"))
        else:
            patience_counter += 1

        if patience_counter == config.PATIENCE:
            print(f"[{model_name} | {dataset_name}] Early stopping, model has not improved for {config.PATIENCE} epochs")
            break

    plot(train_history, val_history, plots_dir, model_name, dataset_name, best_epoch_idx)
    print(f"\n[{model_name} | {dataset_name}] Training completed! Best epoch: {best_epoch_idx}\n")

if __name__ == '__main__':
    total_combinations = len(config.DATASETS) * len(config.MODELS)
    current_combination = 0
    
    print(f"\nStarting training for {len(config.MODELS)} models across {len(config.DATASETS)} datasets")
    print(f"Total combinations: {total_combinations}\n")
    
    for dataset in config.DATASETS:
        for model in config.MODELS:
            current_combination += 1
            print(f"\n*** Progress: {current_combination}/{total_combinations} ***")
            try:
                train_model(model, dataset)
            except Exception as e:
                print(f"ERROR training {model} on {dataset}: {e}")