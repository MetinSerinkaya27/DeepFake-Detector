import os

def check_dataset_balance():
    base_dir = "../../datasets"
    
    print("--- VERİ SETİ KONTROL RAPORU ---")
    
    for split in ['train', 'val']:
        print(f"\n[{split.upper()} KLASÖRÜ]")
        total = 0
        for label in ['real', 'fake']:
            path = os.path.join(base_dir, split, label)
            if os.path.exists(path):
                count = len(os.listdir(path))
                total += count
                print(f" -> {label.upper()}: {count} resim")
            else:
                print(f" -> {label.upper()}: KLASÖR YOK! ({path})")
        print(f" Toplam {split.upper()} Resmi: {total}")

if __name__ == "__main__":
    check_dataset_balance()