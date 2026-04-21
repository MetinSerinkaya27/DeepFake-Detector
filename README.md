#  DeepFake & Image Forgery Dedektörü

Bu proje, dijital görüntülerdeki sahtecilik (forgery), manipülasyon ve DeepFake izlerini tespit etmek amacıyla geliştirilmiş **yapay zeka destekli** bir analiz laboratuvarıdır. Sistem, hem geleneksel bilgisayarlı görü algoritmalarını hem de son teknoloji Derin Öğrenme (Deep Learning) mimarilerini tek bir çatı altında birleştirir.

## 🚀 Özellikler & Algoritmalar

Sistem iki ana koldan (Geleneksel ve Yapay Zeka) analiz yapabilmektedir:

### 🧠 1. Yapay Zeka (AI) Prototipleri
* **ViT (Vision Transformer):** Google'ın önceden eğittiği model üzerine *Transfer Learning* uygulanarak sahtecilik tespitine (Fine-Tuning) adapte edilmiş Şampiyon modelimiz (**%81.12 Başarı**). Resmin global ışık ve doku uyumunu analiz eder.
* **CNN (Convolutional Neural Network):** Sıfırdan tasarlanmış, BatchNorm ve Dropout ile güçlendirilmiş derin özellik çıkarıcı model. Lokal piksel dokusu ve gürültü (noise) anomalilerini tespit eder (**%76.54 Başarı**).
* **LSTM (Long Short-Term Memory):** Görüntüyü 16 farklı parçaya (patch) bölerek pikseller arası akış ve dizilim bozukluklarını yakalayan sekans modelimiz.
* **👑 VOLTRON AI (Ensemble Learning):** Sistemin en güçlü silahı! Görüntüyü ViT, CNN ve LSTM modellerine aynı anda sokup, her birinin güven skorunu hesaplayarak ortak bir konsorsiyum kararı (Ağırlıklı Ortalama) üretir.

### 🕵️‍♂️ 2. Geleneksel Bilgisayarlı Görü Filtreleri
Özellikle "Kopyala-Yapıştır" (Copy-Move Forgery) sahteciliklerini bulmak için kullanılan geleneksel dedektörler:
* **SIFT & SURF:** Yüksek hassasiyetli anahtar nokta eşleştirme.
* **AKAZE & ORB:** Yüksek performanslı ve hızlı klon tespiti.

---

## 🛠️ Kullanılan Teknolojiler

* **Backend (Çekirdek):** Python, FastAPI, Uvicorn
* **Makine Öğrenmesi:** PyTorch, Torchvision, OpenCV, Scikit-Learn
* **Frontend (Arayüz):** React, Vite, Tailwind CSS, Lucide React

---

## 💻 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

### 1. Backend (Yapay Zeka Sunucusu) Kurulumu
Python 3.10+ yüklü olduğundan emin olun.
```bash
# Proje dizinine gidin
cd Forensic-Lab-DeepFake-Detector

# Gerekli kütüphaneleri kurun
pip install fastapi uvicorn python-multipart torch torchvision opencv-python Pillow numpy

# Sunucuyu başlatın (Port: 8000)
uvicorn main:app --reload
# Frontend dizinine gidin (Eğer projeyi frontend klasörüne kurduysanız)
cd frontend

# Bağımlılıkları yükleyin
npm install

# Arayüzü başlatın
npm run dev

⚠️ Önemli Not (Model Ağırlıkları Hakkında)
GitHub'ın 100 MB dosya boyutu sınırı nedeniyle, eğitilmiş yapay zeka ağırlık dosyaları (.pth uzantılı dosyalar) bu depoya yüklenmemiştir.

Projeyi lokalinizde AI algoritmalarıyla çalıştırmak isterseniz:

İlgili modelleri (train_cnn.py, train_vit.py, train_lstm.py) kendi veri setinizle yeniden eğitebilirsiniz.

Eğitim sonrası oluşan cnn_forgery_best.pth, vit_forgery_best.pth ve lstm_forgery_best.pth dosyalarını src/models/ dizini içine yerleştirmelisiniz.

Geliştirici: Metin Serinkaya | 2026