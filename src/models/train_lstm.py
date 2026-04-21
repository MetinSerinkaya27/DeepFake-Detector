import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_models import LSTMDetector

def train_lstm_final():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dengeli LSTM Uzun Dönem Eğitimi Başlıyor (30 Epoch). Cihaz: {device}")

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(root='../../datasets/train', transform=transform)
    val_data = datasets.ImageFolder(root='../../datasets/val', transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = LSTMDetector(input_size=2352, hidden_size=128).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    best_val_acc = 0.0

    for epoch in range(30): # Epoch 30'a çıkarıldı
        # --- EĞİTİM ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            patches = []
            for i in range(4):
                for j in range(4):
                    patch = images[:, :, i*28:(i+1)*28, j*28:(j+1)*28]
                    patches.append(patch.reshape(images.size(0), -1))
            
            input_seq = torch.stack(patches, dim=1).to(device)
            labels = labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(input_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Eğitim Başarısını (Train Acc) Hesapla
            predicted_train = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted_train == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # --- VALIDATION (Analizli) ---
        model.eval()
        val_correct, val_total = 0, 0
        fake_pred, real_pred = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                patches_v = []
                for i in range(4):
                    for j in range(4):
                        p = images[:, :, i*28:(i+1)*28, j*28:(j+1)*28]
                        patches_v.append(p.reshape(images.size(0), -1))
                
                input_seq_v = torch.stack(patches_v, dim=1).to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(input_seq_v)
                predicted = (outputs > 0.5).float()
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                fake_pred += (predicted == 0).sum().item()
                real_pred += (predicted == 1).sum().item()

        val_acc = 100 * val_correct / val_total
        
        # Her iki başarı oranını yan yana yazdırıyoruz
        print(f"Epoch [{epoch+1}/30] | Loss: {train_loss/len(train_loader):.4f} | Train Acc: %{train_acc:.2f} | Val Acc: %{val_acc:.2f}")
        print(f"   [Analiz] Sınav Tahminleri -> GERÇEK: {real_pred} | SAHTE: {fake_pred}")

        # Sadece Val Acc artarsa kaydet (Kusursuz Koruma)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "lstm_forgery_best.pth")
            print(f"--> [YENİ REKOR] En iyi model kaydedildi!")

    print(f"Eğitim bitti. Ulaşılan en yüksek doğrulama başarısı: %{best_val_acc:.2f}")

if __name__ == "__main__":
    train_lstm_final()