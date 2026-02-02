import torch
import os
import sys
import glob
from torch.utils.data import DataLoader, Dataset

# --- SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Konfigurasi Khusus Hewan
MODEL_PATH = os.path.join(BASE_DIR, "animal_final.pth")
DATA_DIR = os.path.join(BASE_DIR, "data_hewan", "labeled")

from src.utils import preprocess_audio
from src.model import ESC50Model

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Mengevaluasi Model Hewan di {device}...")

    # 1. Cek Data
    if not os.path.exists(DATA_DIR):
        print("❌ Data tidak ditemukan."); return
    
    CLASSES = sorted(os.listdir(DATA_DIR))
    print(f"ℹ️  Kelas: {CLASSES}")
    
    # 2. Load Model
    model = ESC50Model(n_classes=len(CLASSES), mode='finetune').to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("❌ Model belum dilatih."); return

    # 3. Kumpulkan Semua File Test
    all_files = []
    cls_map = {k: v for v, k in enumerate(CLASSES)}
    
    for cat in CLASSES:
        files = glob.glob(os.path.join(DATA_DIR, cat, "*.wav"))
        for f in files:
            all_files.append((f, cls_map[cat]))
            
    print(f"📂 Menguji {len(all_files)} file audio...")

    # 4. Loop Pengujian
    correct = 0
    total = len(all_files)
    
    with torch.no_grad():
        for i, (f_path, label_idx) in enumerate(all_files):
            img = preprocess_audio(f_path).unsqueeze(0).to(device)
            label = torch.tensor([label_idx]).to(device)
            
            out = model(img)
            
            if out.dim() == 1: out = out.unsqueeze(0)
                
            pred = out.argmax(1)
            
            if pred.item() == label_idx:
                correct += 1
            
            if (i+1) % 10 == 0:
                print(f"⏳ Progress: {i+1}/{total}...", end="\r")

    # 5. Hasil Akhir
    acc = (correct / total) * 100
    print(f"\n\n✅ SELESAI!")
    print(f"🏆 Akurasi Total: {acc:.2f}% ({correct}/{total} Benar)")

if __name__ == "__main__":
    evaluate()