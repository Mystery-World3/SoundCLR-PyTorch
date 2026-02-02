import torch
import torch.nn.functional as F
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "animal_final.pth")
LABELED_DIR = os.path.join(BASE_DIR, "data_hewan", "labeled")

from src.utils import preprocess_audio
from src.model import ESC50Model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*40)
    print("   DETEKSI SUARA HEWAN")
    print("="*40)
    
    if not os.path.exists(LABELED_DIR):
        print("❌ Error: Folder data hewan tidak ditemukan.")
        return
        
    CLASSES = sorted(os.listdir(LABELED_DIR))
    print(f"ℹ️  Target Hewan: {', '.join(CLASSES)}")
    
    # Load Model
    model = ESC50Model(n_classes=len(CLASSES), mode='finetune').to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ Model Hewan Siap!")
    else:
        print("❌ Model belum di-training.")
        return
    
    while True:
        try:
            path = input("\n🎤 Masukkan Path Audio (.wav) [x=exit]: ").strip().strip('"').strip("'")
            if path.lower() == 'x': break
            if not os.path.exists(path): print("❌ File tidak ada."); continue
            
            # Predict
            inp = preprocess_audio(path).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
                if out.dim() == 1: out = out.unsqueeze(0)
                probs = F.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                
            print(f"🐾 Prediksi: {CLASSES[idx.item()].upper()} ({conf.item()*100:.1f}%)")
            
        except Exception as e:
            print(f"Error: {e}")
            
if __name__ == "__main__":
    main()