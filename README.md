# Dashboard & Rapor Analizörü — v6.1 (UI düzenli)
- Üç sağlayıcı: OpenAI, Azure OpenAI, Google Gemini
- Sekmeli arayüz: Bağlantı → Yükle → Sonuçlar
- Kart tasarımı, status/ilerleme, kompakt yerleşim

## Çalıştırma
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```