import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

import config
from dataset import get_dataset
from models import get_model


def load_model(model_name, dataset_name, out_channels, device):
    """Load a trained model from weights"""
    model = get_model(model_name, imgChannels=3, outChannels=out_channels).to(device)
    
    weights_path = os.path.join("..", "weights", model_name, dataset_name, 
                                f"{model_name}_{dataset_name}_epoch_best.pth")
    
    if not os.path.exists(weights_path):
        print(f"Warning: Weights not found for {model_name} on {dataset_name}")
        return None
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict(model, image, device, task='binary'):
    """Get model prediction for a single image"""
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image)
        
        if task == 'binary':
            # Apply sigmoid and threshold for binary segmentation
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()
        else:
            # Apply softmax and argmax for multiclass segmentation
            pred = torch.softmax(output, dim=1)
            pred = torch.argmax(pred, dim=1, keepdim=True).float()
        
        return pred.squeeze(0).cpu()  # Remove batch dimension


def visualize_predictions(dataset_name):
    """
    Visualize predictions from all active models on the first test sample
    
    Args:
        dataset_name: Name of the dataset to visualize
    """
    sample_idx = 0  # Always use the first image
    device = config.DEVICE
    
    # Get dataset and metadata
    test_ds, metadata = get_dataset(dataset_name, split='test', download=False)
    
    # Get sample
    image, mask = test_ds[sample_idx]
    
    # Load all active models
    predictions = {}
    for model_name in config.MODELS:
        model = load_model(model_name, dataset_name, metadata['out_channels'], device)
        if model is not None:
            pred = predict(model, image, device, metadata['task'])
            predictions[model_name] = pred
    
    # Calculate layout - all rows will have the same number of columns
    num_models = len(predictions)
    # Maximum width is 2 items (original + ground truth) or number of models, whichever is larger
    num_cols = max(2, min(4, num_models))
    
    # Calculate number of rows needed for models
    model_rows = (num_models + num_cols - 1) // num_cols  # Ceiling division
    num_rows = 1 + model_rows  # 1 row for original/ground truth + rows for models
    
    # Create figure
    fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
    
    # Convert tensors to numpy for display
    # Image: (C, H, W) -> (H, W, C)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Mask: (C, H, W) -> (H, W) for binary, keep for multiclass
    if metadata['task'] == 'binary':
        mask_np = mask.squeeze().cpu().numpy()
    else:
        mask_np = mask.squeeze().cpu().numpy()
    
    # First row: Original Image and Ground Truth
    start_pos = (num_cols - 2) / 2.0
    
    # Original Image
    ax1 = plt.subplot(num_rows, num_cols, int(start_pos) + 1)
    ax1.imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Ground Truth
    ax2 = plt.subplot(num_rows, num_cols, int(start_pos) + 2)
    if metadata['task'] == 'binary':
        ax2.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
    else:
        ax2.imshow(mask_np, cmap='tab20')
    ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Model Predictions
    model_list = list(predictions.items())
    
    for row_idx in range(model_rows):
        # Calculate how many items in this row
        start_idx = row_idx * num_cols
        end_idx = min(start_idx + num_cols, num_models)
        items_in_row = end_idx - start_idx
        
        # Calculate starting position to center items in this row
        row_start_pos = (num_cols - items_in_row) / 2.0
        
        for col_idx in range(items_in_row):
            global_idx = start_idx + col_idx
            model_name, pred = model_list[global_idx]
            
            # Calculate subplot position
            subplot_idx = (row_idx + 1) * num_cols + int(row_start_pos) + col_idx + 1
            ax = plt.subplot(num_rows, num_cols, subplot_idx)
            
            pred_np = pred.squeeze().cpu().numpy()
            
            if metadata['task'] == 'binary':
                ax.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(pred_np, cmap='tab20')
            
            ax.set_title(f'{model_name} Prediction', fontsize=12, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle(f'Segmentation Results - {dataset_name}', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    save_dir = os.path.join("..", "visualizations", dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"predictions_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize UNet model predictions')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Dataset name (e.g., CovidQUExMSBench). If not provided, visualizes all datasets in config.')
    
    args = parser.parse_args()
    
    # If no dataset specified, use all datasets from config
    if args.dataset is None:
        print(f"No dataset specified. Visualizing all {len(config.DATASETS)} datasets from config...\n")
        for dataset in config.DATASETS:
            print(f"\n{'='*80}")
            print(f"Processing dataset: {dataset}")
            print(f"{'='*80}")
            try:
                visualize_predictions(dataset)
            except Exception as e:
                print(f"Error visualizing {dataset}: {e}")
    else:
        visualize_predictions(args.dataset)
