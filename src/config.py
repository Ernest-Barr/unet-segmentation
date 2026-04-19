COVID_FILE_PATH = "data/covidquex_256.npz"
COVID_IN_CHANNELS = 1
COVID_OUT_CHANNELS = 1

CELLNUCLEI_FILE_PATH = "data/cellnuclei_256.npz"
CELLNUCLEI_IN_CHANNELS = 3
CELLNUCLEI_OUT_CHANNELS = 1

NUSET_FILE_PATH = "data/NUSET_256.npz"
NUSET_IN_CHANNELS = 1
NUSET_OUT_CHANNELS = 1

BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 10
DELTA_PATIENCE = 0.005 # TODO: Verify number
NUM_WORKERS = 4
DEVICE = "cuda"

DATASET_PATH = ""
WEIGHTS_PATH = ""
RESULTS_PATH = ""

DATASETS = {

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
