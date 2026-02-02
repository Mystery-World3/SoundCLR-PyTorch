import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# --- SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FOLDER = os.path.join(BASE_DIR, "ESC-50-master", "audio")
CSV_FILE = os.path.join(BASE_DIR, "ESC-50-master", "meta", "esc50.csv")

DATA_DIR = os.path.join(BASE_DIR, "data_hewan")

TARGET_CLASSES = [
    'dog', 'rooster', 'pig', 'cow', 'frog', 
    'cat', 'hen', 'insects', 'sheep', 'crow'
]

def prepare():
    print(f"🔍 Memfilter dataset khusus HEWAN...")
    
    # Cek Source
    if not os.path.exists(CSV_FILE):
        print("❌ Error: Folder ESC-50-master tidak ditemukan.")
        return

    df = pd.read_csv(CSV_FILE)
    
    df_hewan = df[df['category'].isin(TARGET_CLASSES)]
    
    print(f"📊 Total Data: {len(df_hewan)} file (dari total 2000)")
    
    # Split Semi-Supervised (80% Unlabeled, 20% Labeled)
    train, test = train_test_split(df_hewan, test_size=0.2, stratify=df_hewan['category'], random_state=42)
    
    # Reset Folder Tujuan
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    
    # 1. menyiapkan Unlabeled
    path_un = os.path.join(DATA_DIR, "unlabeled")
    os.makedirs(path_un, exist_ok=True)
    
    print(f"📦 Menyalin {len(train)} file ke Unlabeled...")
    for fname in train['filename']:
        src = os.path.join(RAW_FOLDER, fname)
        dst = os.path.join(path_un, fname)
        if os.path.exists(src): shutil.copy(src, dst)
        
    # 2. menyiapkan Labeled
    path_lab = os.path.join(DATA_DIR, "labeled")
    print(f"🏷️  Menyalin {len(test)} file ke Labeled...")
    
    for idx, row in test.iterrows():
        cat = row['category']
        fname = row['filename']
        
        cat_dir = os.path.join(path_lab, cat)
        os.makedirs(cat_dir, exist_ok=True)
        
        src = os.path.join(RAW_FOLDER, fname)
        dst = os.path.join(cat_dir, fname)
        if os.path.exists(src): shutil.copy(src, dst)

    print("\n✅ Siap! Folder 'data_hewan' berhasil dibuat.")

if __name__ == '__main__':
    prepare()