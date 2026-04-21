import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_models import CNNDetector

def train_cnn_ultimate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ULTIMATE CNN Eğitimi Başlıyor (30 Epoch - Scheduler Aktif). Cihaz: {device}")

    # 1. YENİ: Işık ve Kontrast oyunları eklendi (ColorJitter)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # <-- Sahtecilik izlerini buldurur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root='../../datasets/train', transform=train_transform)
    val_data = datasets.ImageFolder(root='../../datasets/val', transform=val_transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = CNNDetector().to(device)
    criterion = nn.BCELoss()
    
    # 2. YENİ: Weight Decay (L2) eklendi (Ezberlemeyi yıkar)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 

    # 3. YENİ: Öğrenme Hızı Planlayıcısı (Val Acc tıkanırsa hızı düşürür)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3) 

    best_val_acc = 0.0

    for epoch in range(30): # Epoch'u 30'a çıkardık çünkü yavaşlayarak öğrenecek
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
        
        print(f"Epoch [{epoch+1}/30] | Loss: {train_loss/len(train_loader):.4f} | Train Acc: %{train_acc:.2f} | Val Acc: %{val_acc:.2f}")
        print(f"   [Analiz] Sınav -> GERÇEK: {real_pred} | SAHTE: {fake_pred}")

        # YENİ: Scheduler'a güncel Val Acc değerini veriyoruz ki tıkanıp tıkanmadığını anlasın
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "cnn_forgery_best.pth")
            print(f"--> [YENİ REKOR] Ultimate model kaydedildi: %{val_acc:.2f}")

    print(f"Eğitim bitti. Ulaşılan en yüksek doğrulama başarısı: %{best_val_acc:.2f}")

if __name__ == "__main__":
    train_cnn_ultimate()