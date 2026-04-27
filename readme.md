# Analysis of UNet Architectures for Medical Image Segmentation 

This repository provides a standardized benchmark for analyzing the performance of various Unet architectures against dataests from MedSegBench. 

## Requirements & Setup

### Environment Setup 
 
```bash
conda create -n unet python=3.10.9
conda activate unet
pip install -r requirements 
```

### Running Training/Testing

#### Datasets

Unfortunately, the automatic download of the datasets through the MedSegBench python package does not seem to work. To alleviate this please download the data sets from the [official zenodo page](https://zenodo.org/records/13358372) and paste the desired datasets that are selected in [config.py](src/config.py) in the data/ directory of the root folder.

#### Setting up the Config
The following lines in [config.py](src/config.py) can be configured to change the datasets/models used in the experiment. 
All models from MedSegBench are available and their names can be found in [dataset.py](src/dataset.py), but their respective weights still need to be downloaded from zenodo.


```python
DATASETS = {
        'CovidQUExMSBench',
        'AbdomenUSMSBench',
        'NucleiMSBench',
        'BusiMSBench',
        'DynamicNuclearMSBench'
}

MODELS = [
        'UNet',
        'AttentionUNet',
        'ResUNet',
        'ResAttentionUNet',
        'DenseUNet',
        'DenseAttentionUNet',
        'UNetPP'
]
```

#### Training & Testing

**Training:**
```bash
cd src/
python train.py 
```

**Testing:**
```bash
cd src/ 
python test.py 
```

**Visualize**
```bash 
cd src/
python visualize.py 
```

## Datasets

All datasets from MedSegBench are supported, their names can be found in the [config](src/config.py). 

## Models

The following models are currently implemented:
- 
- U-Net
- U-Net++ 
- Attention U-Net
- Dense U-Net
- Residual U-Net
- Residual Attention U-Net 
- Dense Attention U-Net


