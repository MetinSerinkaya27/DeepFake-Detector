import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_models import ViTDetector

def train_vit_final():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👑 ViT (Vision Transformer) Transfer Öğrenme Başlıyor (10 Epoch). Cihaz: {device}")

    # ViT önceden eğitildiği için Augmentation (döndürme vs) yapmaya pek gerek yoktur
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root='../../datasets/train', transform=transform)
    val_data = datasets.ImageFolder(root='../../datasets/val', transform=transform)
    
    # Batch Size 32'de ekran kartı (CUDA) hafıza hatası verirse, burayı 16 yapabilirsin
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = ViTDetector().to(device)
    criterion = nn.BCELoss()
    
    # ÇOK ÖNEMLİ: ViT'in eski zekasını silmemek için hızı (LR) inanılmaz düşük tutuyoruz
    optimizer = optim.Adam(model.parameters(), lr=0.00003) 

    best_val_acc = 0.0

    for epoch in range(10): # Zaten doktoralı olduğu için 10 tur fazlasıyla yeter
        # --- EĞİTİM ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            predicted_train = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted_train == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # --- VALIDATION ---
        model.eval()
        val_correct, val_total = 0, 0
        fake_pred, real_pred = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                fake_pred += (predicted == 0).sum().item()
                real_pred += (predicted == 1).sum().item()

        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/10] | Loss: {train_loss/len(train_loader):.4f} | Train Acc: %{train_acc:.2f} | Val Acc: %{val_acc:.2f}")
        print(f"   [Analiz] Sınav -> GERÇEK: {real_pred} | SAHTE: {fake_pred}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "vit_forgery_best.pth")
            print(f"--> [ŞAMPİYON] En iyi ViT modeli kaydedildi: %{val_acc:.2f}")

    print(f"ViT Eğitimi bitti. Ulaşılan en yüksek başarı: %{best_val_acc:.2f}")

if __name__ == "__main__":
    train_vit_final()