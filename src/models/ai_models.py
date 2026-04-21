import torch
import torch.nn as nn
import torchvision.models as models # ViT için bu kütüphane şart
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# --- 1. CNN Mimarisi (Hiperparametreleri Güçlendirilmiş V2) ---
class CNNDetector(nn.Module):
    def __init__(self):
        super(CNNDetector, self).__init__()
        self.features = nn.Sequential(
            # 1. Katman
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), # Öğrenmeyi dengeler
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112x112

            # 2. Katman
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 56x56
            
            # 3. Katman (Ekstra Derinlik)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Ezberlemeyi (Overfitting) engeller
            nn.Linear(128 * 28 * 28, 256), 
            nn.ReLU(),
            nn.Dropout(0.3), # Son çıkıştan önce ufak bir zorlama daha
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# --- 2. LSTM Mimarisi: Resim blokları (16-Patch) arasındaki akış bozukluğuna bakar ---
class LSTMDetector(nn.Module):
    def __init__(self, input_size=2352, hidden_size=128): # 16 Patch için 2352
        super(LSTMDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, sequence_length=16, features=2352)
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


# --- 3. YENİ: ViT Mimarisi (Global Işık ve Doku Analizi) ---
class ViTDetector(nn.Module):
    def __init__(self):
        super(ViTDetector, self).__init__()
        # Google'ın önceden eğittiği ağırlıkları indiriyoruz
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        # Son katmanı kendi (Sahte/Gerçek) hedefimize göre kesip biçiyoruz
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.vit(x)


# --- Tahmin Fonksiyonları (Backend & React İçin) ---

def predict_with_cnn(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNDetector().to(device)
    
    # En iyi modeli yüklüyoruz
    model_path = os.path.join(os.path.dirname(__file__), "cnn_forgery_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[UYARI] CNN model dosyası bulunamadı: {model_path}")
    
    model.eval()
    
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(input_tensor).item()
    
    is_fake = prob > 0.5
    return {
        "is_fake": is_fake,
        "confidence": round(prob * 100 if is_fake else (1 - prob) * 100, 2),
        "method": "CNN (Lokal Doku Analizi)"
    }

def predict_with_lstm(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMDetector(input_size=2352).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), "lstm_forgery_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[UYARI] LSTM model dosyası bulunamadı: {model_path}")
        
    model.eval()

    img = Image.open(image_path).convert('RGB').resize((112, 112))
    img_tensor = transforms.ToTensor()(img)
    
    # --- 16 Patch (4x4 Grid) Tahmin Mantığı ---
    patches = []
    for i in range(4):
        for j in range(4):
            # Her parça 28x28 piksel
            patch = img_tensor[:, i*28:(i+1)*28, j*28:(j+1)*28]
            patches.append(patch.flatten())
            
    input_seq = torch.stack(patches).unsqueeze(0).to(device) # (1, 16, 2352)

    with torch.no_grad():
        prob = model(input_seq).item()

    is_fake = prob > 0.5
    return {
        "is_fake": is_fake,
        "confidence": round(prob * 100 if is_fake else (1 - prob) * 100, 2),
        "method": "LSTM (Sekans Analizi)"
    }

def predict_with_vit(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTDetector().to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), "vit_forgery_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[UYARI] ViT model dosyası bulunamadı: {model_path}")
    
    model.eval()
    
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(input_tensor).item()
    
    is_fake = prob > 0.5
    return {
        "is_fake": is_fake,
        "confidence": round(prob * 100 if is_fake else (1 - prob) * 100, 2),
        "method": "ViT (Global Uyum Analizi)"
    }