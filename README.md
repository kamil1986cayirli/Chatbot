# Dashboard & Rapor Analizörü — v6.1.1
- Gemini 404 için model adları `gemini-1.5-flash-001` / `gemini-1.5-pro-001` olarak güncellendi.
- Gerekirse otomatik `models/` öneki fallback’i var.
- UI: 3 sekme + kartlar (v6.1 ile aynı).

## Çalıştırma
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```