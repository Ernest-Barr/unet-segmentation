BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 10
DELTA_PATIENCE = 0.005  # TODO: Verify number
NUM_WORKERS = 4
DEVICE = "cuda"
LEARN_RATE  = 1e-3
DATASET_PATH = ""
WEIGHTS_PATH = ""
RESULTS_PATH = ""

IMAGE_SIZE = 256
DATA_DIR = "../data/"

DATASET_METADATA = {
    'AbdomenUSMSBench': {'modality': 'Ultrasound', 'out_channels': 9, 'task': 'multiclass'},
    'Bbbc010MSBench': {'modality': 'Microscopy', 'out_channels': 1, 'task': 'binary'},
    'Bkai-Igh-MSBench': {'modality': 'Endoscopy', 'out_channels': 3, 'task': 'multiclass'},
    'BriFiSegMSBench': {'modality': 'Microscopy', 'out_channels': 1, 'task': 'binary'},
    'BusiMSBench': {'modality': 'Ultrasound', 'out_channels': 1, 'task': 'binary'},
    'CellnucleiMSBench': {'modality': 'Nuclei', 'out_channels': 1, 'task': 'binary'},
    'ChaseDB1MSBench': {'modality': 'Fundus', 'out_channels': 1, 'task': 'binary'},
    'ChuacMSBench': {'modality': 'Fundus', 'out_channels': 1, 'task': 'binary'},
    'Covid19RadioMSBench': {'modality': 'Chest X-Ray', 'out_channels': 1, 'task': 'binary'},
    'CovidQUExMSBench': {'modality': 'Chest X-Ray', 'out_channels': 1, 'task': 'binary'},
    'CystoFluidMSBench': {'modality': 'OCT', 'out_channels': 1, 'task': 'binary'},
    'Dca1MSBench': {'modality': 'Fundus', 'out_channels': 1, 'task': 'binary'},
    'DeepbacsMSBench': {'modality': 'Microscopy', 'out_channels': 1, 'task': 'binary'},
    'DriveMSBench': {'modality': 'Fundus', 'out_channels': 1, 'task': 'binary'},
    'DynamicNuclearMSBench': {'modality': 'Nuclear Cell', 'out_channels': 1, 'task': 'binary'},
    'FHPsAOPMSBench': {'modality': 'Ultrasound', 'out_channels': 3, 'task': 'multiclass'},
    'IdribMSBench': {'modality': 'Fundus', 'out_channels': 1, 'task': 'binary'},
    'Isic2016MSBench': {'modality': 'Dermoscopy', 'out_channels': 1, 'task': 'binary'},
    'Isic2018MSBench': {'modality': 'Dermoscopy', 'out_channels': 1, 'task': 'binary'},
    'KvasirMSBench': {'modality': 'Endoscopy', 'out_channels': 1, 'task': 'binary'},
    'M2caiSegMSBench': {'modality': 'Endoscopy', 'out_channels': 19, 'task': 'multiclass'},
    'MonusacMSBench': {'modality': 'Pathology', 'out_channels': 6, 'task': 'multiclass'},
    'MosMedPlusMSBench': {'modality': 'CT', 'out_channels': 1, 'task': 'binary'},
    'NucleiMSBench': {'modality': 'Pathology', 'out_channels': 1, 'task': 'binary'},
    'NusetMSBench': {'modality': 'Nuclear Cell', 'out_channels': 1, 'task': 'binary'},
    'PandentalMSBench': {'modality': 'X-Ray', 'out_channels': 1, 'task': 'binary'},
    'PolypGenMSBench': {'modality': 'Endoscopy', 'out_channels': 1, 'task': 'binary'},
    'Promise12MSBench': {'modality': 'MRI', 'out_channels': 1, 'task': 'binary'},
    'RoboToolMSBench': {'modality': 'Endoscopy', 'out_channels': 1, 'task': 'binary'},
    'TnbcnucleiMSBench': {'modality': 'Pathology', 'out_channels': 1, 'task': 'binary'},
    'UltrasoundNerveMSBench': {'modality': 'Ultrasound', 'out_channels': 1, 'task': 'binary'},
    'USforKidneyMSBench': {'modality': 'Ultrasound', 'out_channels': 1, 'task': 'binary'},
    'UWSkinCancerMSBench': {'modality': 'Dermoscopy', 'out_channels': 1, 'task': 'binary'},
    'WbcMSBench': {'modality': 'Microscopy', 'out_channels': 3, 'task': 'multiclass'},
    'YeazMSBench': {'modality': 'Microscopy', 'out_channels': 1, 'task': 'binary'},
}

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
    # 'DenseUNet',
    # 'DenseAttentionUNet',
    # 'UNetPP'
]
