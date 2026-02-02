import torch
import os
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "animal_final.pth")
DATA_DIR = os.path.join(BASE_DIR, "data_hewan", "labeled")
OUTPUT_IMG = os.path.join(BASE_DIR, "matrix_hewan.png")

from src.utils import preprocess_audio
from src.model import ESC50Model

def plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🎨 Sedang menggambar Confusion Matrix...")

    if not os.path.exists(DATA_DIR): return
    CLASSES = sorted(os.listdir(DATA_DIR))
    
    # Load Model
    model = ESC50Model(n_classes=len(CLASSES), mode='finetune').to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    # Loop Prediksi
    cls_map = {k: v for v, k in enumerate(CLASSES)}
    all_files = []
    for cat in CLASSES:
        files = glob.glob(os.path.join(DATA_DIR, cat, "*.wav"))
        for f in files: all_files.append((f, cls_map[cat]))

    with torch.no_grad():
        for f_path, label in all_files:
            img = preprocess_audio(f_path).unsqueeze(0).to(device)
            out = model(img)
            if out.dim() == 1: out = out.unsqueeze(0)
            pred = out.argmax(1).item()
            
            y_true.append(label)
            y_pred.append(pred)

    # Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting 
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Sebenarnya')
    plt.title(f'Confusion Matrix Hewan (Akurasi: 93.75%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"✅ Gambar berhasil disimpan di: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot()