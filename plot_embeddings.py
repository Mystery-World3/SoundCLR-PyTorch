import torch
import torch.nn as nn
import os
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# --- SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Path Data & Model
MODEL_PATH = os.path.join(BASE_DIR, "animal_final.pth")
DATA_DIR = os.path.join(BASE_DIR, "data_hewan", "labeled")
OUTPUT_IMG = os.path.join(BASE_DIR, "embedding_tsne.png")

from src.utils import preprocess_audio
from src.model import ESC50Model

def extract_embeddings_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Memulai ekstraksi fitur di {device}...")

    # 1. Load Daftar Kelas
    if not os.path.exists(DATA_DIR):
        print("❌ Error: Folder data_hewan tidak ditemukan.")
        return
    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])    
    # 2. Load Model
    model = ESC50Model(n_classes=len(CLASSES), mode='finetune').to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("❌ Error: File animal_final.pth belum ada.")
        return

    if hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    else:
        model.fc = nn.Identity() 
    
    # 3. Mengumpulkan Data
    all_files = []
    for cat in CLASSES:
        files = glob.glob(os.path.join(DATA_DIR, cat, "*.wav"))
        for f in files:
            all_files.append((f, cat)) 

    print(f"📂 Memproses {len(all_files)} file audio untuk ekstraksi fitur...")

    embeddings = []
    labels = []

    # 4. Looping Ekstraksi
    with torch.no_grad():
        for i, (f_path, label) in enumerate(all_files):
            # Preprocess
            img = preprocess_audio(f_path).unsqueeze(0).to(device)
            
            # Feed Forward 
            fitur = model(img)
            
            # Ubah ke numpy dan simpan
            embeddings.append(fitur.cpu().numpy().flatten())
            labels.append(label)
            
            if (i+1) % 10 == 0:
                print(f"⏳ Progress: {i+1}/{len(all_files)}...", end="\r")

    embeddings = np.array(embeddings)
    
    # 5. Reduksi Dimensi dengan t-SNE 
    print("\n\n🎨 Menghitung t-SNE (Ini butuh beberapa detik)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 6. Menggambar Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", len(CLASSES)),
        s=100,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5
    )
    
    plt.title('t-SNE Visualization of Audio Feature Embeddings', fontsize=16, pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Kategori Hewan")
    plt.tight_layout()

    # 7. Menyimpan Gambar
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"✅ Selesai! Gambar Visualisasi t-SNE disimpan di: {OUTPUT_IMG}")

if __name__ == "__main__":
    extract_embeddings_and_plot()