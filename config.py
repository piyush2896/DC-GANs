class Config:
    BATCH_SIZE = 128
    IMG_SIZE = 64
    N_CHANNELS = 3
    N_DIMS = 100
    GEN_FEATURE_MAPS = 64
    DIS_FEATURE_MAPS = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 2e-4
    BETA = 0.5
    WORKERS = 2
    ROOT = './data/celeba'
    DEVICE = 'gpu'
    MODEL_PATH = './model_weights'
