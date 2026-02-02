import librosa
import torch
import numpy as np
from .config import Config

def preprocess_audio(file_path):
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=Config.SR, duration=Config.DURATION)
        
        # 2. Fix Length (Padding/Cutting)
        if len(y) < Config.N_SAMPLES:
            padding = Config.N_SAMPLES - len(y)
            y = np.pad(y, (0, padding))
        else:
            y = y[:Config.N_SAMPLES]
            
        # 3. Mel Spectrogram
        spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=Config.N_MELS, n_fft=Config.N_FFT, hop_length=Config.HOP_LEN
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # 4. Normalize 0-1
        spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)
        
        # 5. To Tensor (1, Freq, Time)
        return torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)
        
    except Exception as e:
        print(f"Error file {file_path}: {e}")
        return torch.zeros(1, Config.N_MELS, 216) # Dummy return