class Config:
    SR = 22050
    DURATION = 5
    N_SAMPLES = SR * DURATION
    
    # Spectrogram
    N_MELS = 128
    N_FFT = 2048
    HOP_LEN = 512
    
    # Training
    BATCH_SIZE = 32
    LR = 3e-4
    EPOCHS_PRE = 50   # Pre-training
    EPOCHS_FINE = 30  # Fine tuning
    NUM_CLASSES = 50  # ESC-50 has 50 classes