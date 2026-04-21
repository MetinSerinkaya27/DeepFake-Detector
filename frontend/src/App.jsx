import React, { useState } from 'react';
import { Upload, Shield, Search, XCircle, Terminal, Activity } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [stats, setStats] = useState(null);
  const [algo, setAlgo] = useState('VOLTRON'); // Varsayılanı en havalısı yaptık
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState(["[SİSTEM] Tüm ajanlar çevrimiçi. Sistem hazır."]);

  const addLog = (m) => setLogs(p => [`> ${m}`, ...p].slice(0, 5));

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setStats(null);
    addLog(`${algo} protokolü başlatıldı...`);

    const fd = new FormData();
    fd.append('file', file);
    fd.append('algorithm', algo);

    try {
      const res = await fetch('http://localhost:8000/analyze', { method: 'POST', body: fd });
      const data = await res.json();

      if (data.status === "wip") {
        setPreview(null);
        alert(data.message);
      } else if (data.resultUrl) {
        // Görüntü güncellemesini engellemek için cache-buster ekliyoruz
        setPreview(`${data.resultUrl}?t=${new Date().getTime()}`);
        setStats(data.stats);
        addLog(`Analiz başarılı: ${data.stats.is_fake ? 'TEHDİT BULUNDU' : 'TEMİZ'}`);
      }
    } catch (e) { 
      addLog("HATA: Sunucu bağlantısı reddedildi."); 
    }
    finally { setLoading(false); }
  };

  return (
    <div className="min-h-screen bg-[#0a0c10] text-slate-300 p-10 font-sans">
      <header className="max-w-7xl mx-auto flex justify-between items-center mb-10 border-b border-slate-800 pb-5">
        <div className="flex items-center gap-3">
          <Shield className="text-red-600" size={30} />
          <h1 className="text-xl font-black tracking-tighter text-white uppercase">Forensic Lab <span className="text-red-600">v2</span></h1>
        </div>
        <div className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Metin Serinkaya // 2026</div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-12 gap-10">
        <div className="col-span-4 space-y-6">
          <div className="bg-[#0d1117] border border-slate-800 rounded-3xl p-8 shadow-xl">
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6 flex items-center gap-2"><Terminal size={14}/> Kontrol Ünitesi</h2>
            
            <select className="w-full bg-[#161b22] border border-slate-700 rounded-xl p-4 mb-6 outline-none focus:border-red-600 font-mono text-xs text-slate-300" value={algo} onChange={e => setAlgo(e.target.value)}>
              <optgroup label="Yapay Zeka (AI) Prototipleri">
                <option value="VOLTRON">VOLTRON (Tüm AI Ajanları)</option>
                <option value="ViT">ViT (Transformer - Şampiyon)</option>
                <option value="CNN">CNN (Derin Doku Analizi)</option>
                <option value="LSTM">LSTM (Sekans ve Akış)</option>
              </optgroup>
              <optgroup label="Geleneksel Filtreler">
                <option value="SIFT">SIFT (Hassas Klon Tespiti)</option>
                <option value="AKAZE">AKAZE (Hızlı Tarama)</option>
                <option value="ORB">ORB (Performans)</option>
              </optgroup>
            </select>
            
            <button onClick={handleAnalyze} disabled={loading || !file} className="w-full bg-red-600 hover:bg-red-700 disabled:bg-slate-800 disabled:text-slate-500 py-4 rounded-xl font-bold text-white transition-all active:scale-95 flex justify-center items-center gap-2">
              {loading ? <><Activity className="animate-spin" size={18}/> TARANIYOR...</> : <><Search size={18}/> ANALİZİ BAŞLAT</>}
            </button>

            {/* SONUÇ RAPORU ALANI */}
            {stats && (
              <div className="mt-8 pt-8 border-t border-slate-800 animate-in fade-in slide-in-from-top duration-500">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Bulgu Raporu</span>
                  <span className={`px-2 py-1 rounded text-[9px] font-bold ${stats.is_fake ? 'bg-red-600 text-white shadow-[0_0_10px_rgba(220,38,38,0.5)]' : 'bg-green-600 text-white'} uppercase`}>
                    {stats.is_fake ? 'Sahte (Tahrifat)' : 'Orijinal (Temiz)'}
                  </span>
                </div>

                {/* EĞER VOLTRON İSE: Özel 3'lü Ajan Görünümü */}
                {algo === "VOLTRON" && stats.details ? (
                  <div className="space-y-3">
                    <div className="bg-black/40 p-4 rounded-lg border border-slate-700/50 flex justify-between items-center">
                      <p className="text-[10px] text-slate-400 uppercase font-bold">Ortak Karar Güveni:</p>
                      <p className={`text-xl font-black ${stats.is_fake ? 'text-red-500' : 'text-green-500'}`}>%{stats.confidence}</p>
                    </div>
                    <div className="grid grid-cols-3 gap-2 mt-2">
                      {['vit', 'cnn', 'lstm'].map(agent => (
                        <div key={agent} className="bg-black/20 p-2 rounded-lg border border-slate-800 flex flex-col items-center justify-center">
                          <p className="text-[8px] text-slate-500 uppercase font-bold mb-1">{agent}</p>
                          <p className={`text-xs font-black ${stats.details[agent].is_fake ? 'text-red-500' : 'text-green-500'}`}>
                            %{stats.details[agent].confidence}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  /* EĞER TEKİL BİR MODEL (SIFT, CNN, ViT vs) İSE */
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-black/20 p-3 rounded-lg border border-slate-800">
                      <p className="text-[9px] text-slate-500 uppercase font-bold">
                        {stats.confidence ? 'Güven Skoru' : 'Klon Nokta'}
                      </p>
                      <p className={`text-lg font-black ${stats.is_fake ? 'text-red-500' : 'text-white'}`}>
                        {stats.confidence ? `%${stats.confidence}` : stats.count}
                      </p>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg border border-slate-800">
                      <p className="text-[9px] text-slate-500 uppercase font-bold">Kullanılan Metot</p>
                      <p className="text-xs font-bold text-slate-300 mt-1 truncate">
                        {stats.method || algo}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="bg-[#0d1117] border border-slate-800 rounded-2xl p-5 font-mono text-[10px] text-green-500/70 h-32 overflow-hidden shadow-inner">
            {logs.map((l, i) => <div key={i} className="mb-1 opacity-80">{l}</div>)}
          </div>
        </div>

        <div className="col-span-8 bg-[#0d1117] border border-slate-800 rounded-[40px] flex items-center justify-center relative min-h-[600px] overflow-hidden shadow-2xl">
          {!preview ? (
            <label className="cursor-pointer group text-center z-10">
              <input type="file" className="hidden" accept="image/*" onChange={e => {
                const s = e.target.files[0];
                if(s) {
                  setFile(s);
                  setPreview(URL.createObjectURL(s));
                  setStats(null);
                  addLog("Hedef dosya yüklendi.");
                }
              }} />
              <div className="w-20 h-20 bg-slate-800/30 rounded-full flex items-center justify-center mx-auto mb-4 border border-slate-700 group-hover:border-red-600 transition-all group-hover:scale-110 group-active:scale-95">
                <Upload className="text-slate-500 group-hover:text-red-600 transition-colors" size={32} />
              </div>
              <p className="text-sm text-slate-500 group-hover:text-slate-300 transition-colors font-mono uppercase tracking-widest">Görsel Seç</p>
            </label>
          ) : (
            <div className="relative p-10 group z-10">
              <img src={preview} alt="Analiz" className="max-w-full max-h-[500px] rounded-2xl shadow-[0_0_30px_rgba(0,0,0,0.5)] border border-slate-700" />
              <button onClick={() => {setFile(null); setPreview(null); setStats(null); addLog("Sistem temizlendi.");}} className="absolute -top-4 -right-4 bg-slate-900 border border-slate-700 p-2 rounded-full text-slate-400 hover:text-red-500 hover:border-red-500 hover:scale-110 transition-all shadow-xl">
                <XCircle size={24} />
              </button>
            </div>
          )}
          {/* Cyberpunk Çerçeveler */}
          <div className="absolute top-8 left-8 w-12 h-12 border-t-2 border-l-2 border-slate-800/50"></div>
          <div className="absolute bottom-8 right-8 w-12 h-12 border-b-2 border-r-2 border-slate-800/50"></div>
          {/* Tarama Efekti (Opsiyonel Estetik) */}
          {loading && <div className="absolute inset-0 bg-red-600/5 animate-pulse z-0"></div>}
        </div>
      </main>
    </div>
  );
}

export default App;