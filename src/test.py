import config
import os
import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from models import get_model


# https://keras.io/api/metrics/segmentation_metrics/
# https://medium.com/mastering-data-science/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f

def calculate_metrics(logits, masks):
    EPS = 1e-9

    probabilities = torch.sigmoid(logits)

    pred_masks = (probabilities > 0.5).float()
    true_masks = (masks > 0.5).float()

    pred_masks_flat = pred_masks.flatten(start_dim=1)
    true_masks_flat = true_masks.flatten(start_dim=1)
    pred_probs_flat = probabilities.flatten(start_dim=1)

    TP = (pred_masks_flat * true_masks_flat).sum(dim=1)
    FP = (pred_masks_flat * (1 - true_masks_flat)).sum(dim=1)
    FN = ((1 - pred_masks_flat) * true_masks_flat).sum(dim=1)
    TN = ((1 - pred_masks_flat) * (1 - true_masks_flat)).sum(dim=1)

    iou = (TP + EPS) / (TP + FP + FN + EPS)
    dice = (2 * TP) / (2 * TP + FP + FN + EPS)
    pixel_acc = (TP + TN + EPS) / (TP + TN + FP + FN + EPS)
    mae = torch.abs(pred_probs_flat - true_masks_flat).mean(dim=1)

    return iou.tolist(), dice.tolist(), pixel_acc.tolist(), mae.tolist()


def test_model(model_name, dataset_name):
    weights_path = os.path.join("..", "weights", model_name, dataset_name,
                                f"{model_name}_{dataset_name}_epoch_best.pth")
    results_dir = os.path.join("..", "results", model_name, dataset_name)
    results_csv = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.csv")

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Skipping {model_name} on {dataset_name}: Weights not found at {weights_path}")
        return

    try:
        test_ds = get_dataset(dataset_name, split="test", download=False)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    except Exception as e:
        print(f"Skipping {dataset_name}: Could not load test dataset. Error: {e}")
        return

    model = get_model(model_name, imgChannels=3, outChannels=1).to(config.DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
    model.eval()

    results = []
    idx = 0

    progress_bar = tqdm(test_loader, desc=f"Testing {model_name} on {dataset_name}", leave=False)

    with torch.no_grad():
        for images, masks in progress_bar:
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

            output = model(images)
            iou, dice, acc, mae = calculate_metrics(output, masks)

            for i in range(len(iou)):
                results.append([
                    model_name,
                    dataset_name,
                    idx,
                    iou[i],
                    dice[i],
                    acc[i],
                    mae[i]
                ])
                idx += 1

    with open(results_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "dataset_name", "image_index", "iou", "dice", "acc", "mae"])
        writer.writerows(results)


if __name__ == '__main__':
    print("Starting Tests:")
    for dataset in config.DATASETS:
        for model in config.MODELS:
            try:
                test_model(model, dataset)
            except Exception as e:
                print(e)
