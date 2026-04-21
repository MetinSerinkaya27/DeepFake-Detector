import os, uuid, shutil, sys
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Proje dizinini yola ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Algoritmaları içeri aktar
from algorithms.traditional_detectors import sahtecilik_yakala_web
# YENİ: predict_with_vit eklendi
from models.ai_models import predict_with_cnn, predict_with_lstm, predict_with_vit 

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "data"
if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
app.mount("/data", StaticFiles(directory=UPLOAD_DIR), name="data")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), algorithm: str = Form(...)):
    try:
        # 1. Dosya Hazırlığı
        ext = file.filename.split(".")[-1]
        uid = str(uuid.uuid4())
        in_p = os.path.join(UPLOAD_DIR, f"in_{uid}.{ext}")
        out_f = f"out_{uid}.{ext}"
        out_p = os.path.join(UPLOAD_DIR, out_f)

        # Gelen dosyayı kaydet
        with open(in_p, "wb") as f: 
            shutil.copyfileobj(file.file, f)

        stats = {}
        
        # --- AI ALGORİTMALARI (User Story-3) ---
        if algorithm == "CNN":
            stats = predict_with_cnn(in_p)
            shutil.copy(in_p, out_p) 
            
        elif algorithm == "LSTM":
            stats = predict_with_lstm(in_p)
            shutil.copy(in_p, out_p)
            
        elif algorithm == "ViT":
            stats = predict_with_vit(in_p)
            shutil.copy(in_p, out_p)

        # YENİ: VOLTRON (3'lü Konsorsiyum)
        elif algorithm == "VOLTRON":
            res_cnn = predict_with_cnn(in_p)
            res_lstm = predict_with_lstm(in_p)
            res_vit = predict_with_vit(in_p)
            
            # Ortak karar için skorları sahtelik oranına göre 0-100 arası hizalıyoruz
            def get_fake_score(res):
                return res["confidence"] if res["is_fake"] else (100 - res["confidence"])
            
            avg_fake = (get_fake_score(res_cnn) + get_fake_score(res_lstm) + get_fake_score(res_vit)) / 3.0
            is_fake = avg_fake > 50
            final_conf = avg_fake if is_fake else (100 - avg_fake)
            
            stats = {
                "is_fake": is_fake,
                "confidence": round(final_conf, 2),
                "method": "VOLTRON AI",
                "details": { "vit": res_vit, "cnn": res_cnn, "lstm": res_lstm }
            }
            shutil.copy(in_p, out_p)

        # --- GELENEKSEL ALGORİTMALAR (User Story-2) ---
        else:
            stats = sahtecilik_yakala_web(in_p, out_p, algorithm)

        # 3. Sonuç Döndürme
        return {
            "status": "success",
            "resultUrl": f"http://localhost:8000/data/{out_f}",
            "stats": stats,
            "message": f"{algorithm} analizi başarıyla tamamlandı."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)