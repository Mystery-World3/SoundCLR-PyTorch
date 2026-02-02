import torch
import glob
import os
import sys
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import Config
from src.utils import preprocess_audio
from src.model import ESC50Model
from src.loss import NTXentLoss

UNLABELED_DIR = os.path.join(BASE_DIR, "data_hewan", "unlabeled")
SAVE_PATH = os.path.join(BASE_DIR, "animal_pretrained.pth")

class UnlabeledDataset(Dataset):
    def __init__(self):
        self.files = glob.glob(os.path.join(UNLABELED_DIR, "*.wav"))
        print(f"📊 Dataset Unlabeled (Hewan): {len(self.files)} file.")
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        return preprocess_audio(f), preprocess_audio(f)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Pre-training Hewan di {device}...")
    
    ds = UnlabeledDataset()
    if len(ds) == 0: 
        print("❌ Data kosong! Jalankan prepare_animals.py dulu."); return
    
    dl = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    model = ESC50Model(mode='pretrain').to(device)
    crit = NTXentLoss()
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    model.train()
    for ep in range(Config.EPOCHS_PRE):
        total_loss = 0
        for v1, v2 in dl:
            v1, v2 = v1.to(device), v2.to(device)
            opt.zero_grad()
            loss = crit(model(v1), model(v2))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        print(f"Epoch {ep+1}/{Config.EPOCHS_PRE} | Loss: {total_loss/len(dl):.4f}")
        
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Pretraining Selesai! Disimpan: {SAVE_PATH}")

if __name__ == '__main__':
    train()