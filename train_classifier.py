import torch
import torch.nn as nn
import glob
import os
import sys
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

PRETRAINED_PATH = os.path.join(BASE_DIR, "animal_pretrained.pth")
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "animal_final.pth")
LABELED_DIR = os.path.join(BASE_DIR, "data_hewan", "labeled")

from src.config import Config
from src.utils import preprocess_audio
from src.model import ESC50Model

# Cek Folder
if not os.path.exists(LABELED_DIR):
    print("❌ Folder data_hewan belum ada. Jalankan prepare_animals.py")
    sys.exit()

CLASSES = sorted(os.listdir(LABELED_DIR))
CLS_MAP = {k: v for v, k in enumerate(CLASSES)}

class LabeledDataset(Dataset):
    def __init__(self):
        self.data = []
        for cat in CLASSES:
            files = glob.glob(os.path.join(LABELED_DIR, cat, "*.wav"))
            for f in files:
                self.data.append((f, CLS_MAP[cat]))
        print(f"📊 Dataset Labeled (Hewan): {len(self.data)} file (Target: {len(CLASSES)} Kelas).")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        f, l = self.data[idx]
        return preprocess_audio(f), torch.tensor(l, dtype=torch.long)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Fine-Tuning Hewan di {device}...")
    
    ds = LabeledDataset()
    dl = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Init Model
    model = ESC50Model(n_classes=len(CLASSES), mode='finetune').to(device)
    
    # Load Pretrain
    if os.path.exists(PRETRAINED_PATH):
        print(f"📥 Memuat otak pre-trained: {os.path.basename(PRETRAINED_PATH)}")
        
        state_dict = torch.load(PRETRAINED_PATH, map_location=device)
        
        cleaned_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
        
        model.load_state_dict(cleaned_state_dict, strict=False)
        print("✅ Berhasil memuat Encoder (Classifier lama dibuang & di-reset).")
    else:
        print("⚠️ Warning: File pretrain tidak ada. Training dari nol.")
    
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    model.train()
    for ep in range(Config.EPOCHS_FINE):
        ok, tot = 0, 0
        for img, lbl in dl:
            img, lbl = img.to(device), lbl.to(device)
            opt.zero_grad()
            out = model(img)
            loss = crit(out, lbl)
            loss.backward()
            opt.step()
            
            ok += (out.argmax(1) == lbl).sum().item()
            tot += lbl.size(0)
            
        print(f"Epoch {ep+1}/{Config.EPOCHS_FINE} | Akurasi: {ok/tot*100:.2f}%")
        
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"✅ Model Final Hewan Selesai! Disimpan: {FINAL_MODEL_PATH}")

if __name__ == '__main__':
    train()