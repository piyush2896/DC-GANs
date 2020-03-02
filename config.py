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
    NUM_WORKERS = 2
    DATA_ROOT = './data/'
    DEVICE = 'cuda'
    MODEL_PATH = './model_weights'
    LOG_STEP = 50
    IMG_LOG_STEP = 500
