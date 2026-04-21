import os
import shutil
import random
from sklearn.model_selection import train_test_split

# --- AYARLAR ---
raw_data_path = r"C:\Users\27met\OneDrive\Masaüstü\CASIA2" 
base_dir = "../../datasets" 

def setup_structure():
    # Eski klasörleri temizle ki üst üste binmesin
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(base_dir, split, label), exist_ok=True)

def organize_dataset():
    setup_structure()
    
    au_folder = os.path.join(raw_data_path, "Au")
    tp_folder = os.path.join(raw_data_path, "Tp")
    
    # --- KRİTİK DÜZELTME: .tif ve .tiff eklendi ---
    valid_ext = ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff')
    authentic = [f for f in os.listdir(au_folder) if f.lower().endswith(valid_ext)]
    tampered = [f for f in os.listdir(tp_folder) if f.lower().endswith(valid_ext)]

    print(f"Filtre Sonrası -> Gerçek (Au): {len(authentic)} | Sahte (Tp): {len(tampered)}")

    # --- EŞİTLEME (BALANCING) ---
    # Hangi sınıf daha azsa, diğerini o sayıya çekiyoruz (%50-%50)
    min_count = min(len(authentic), len(tampered))
    
    # Resimleri karıştırıp sadece min_count kadar alıyoruz
    random.shuffle(authentic)
    random.shuffle(tampered)
    
    authentic = authentic[:min_count]
    tampered = tampered[:min_count]
    
    print(f"Eşitleme Yapıldı -> Eğitime Girecek: {min_count} Gerçek, {min_count} Sahte")

    # %80 Eğitim, %20 Doğrulama
    train_au, val_au = train_test_split(authentic, test_size=0.2, random_state=42)
    train_tp, val_tp = train_test_split(tampered, test_size=0.2, random_state=42)

    def copy_files(files, source_subdir, split, label):
        source_base = os.path.join(raw_data_path, source_subdir)
        target_path = os.path.join(base_dir, split, label)
        for f in files:
            shutil.copy(os.path.join(source_base, f), os.path.join(target_path, f))

    print("Dosyalar kopyalanıyor, lütfen bekleyin...")
    copy_files(train_au, "Au", 'train', 'real')
    copy_files(val_au, "Au", 'val', 'real')
    copy_files(train_tp, "Tp", 'train', 'fake')
    copy_files(val_tp, "Tp", 'val', 'fake')
    
    print("İşlem Başarıyla Tamamlandı! Veri seti artık %100 Dengeli ve Kusursuz.")

if __name__ == "__main__":
    organize_dataset()