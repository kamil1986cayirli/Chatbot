# Dashboard & Rapor Analizörü — v5.3 (Cloud Pack)

**Özellikler:** Görsel (PNG/JPG), PDF, CSV/TSV, Excel (XLSX/XLS), Parquet analizi; JSON çıktı; KPI kartları; CSV/Markdown dışa aktarım; aynı pencerede Q&A; sırayla analiz + ilerleme çubuğu.

## Hızlı Başlangıç (Lokal)
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

## Streamlit Cloud
- `OPENAI_API_KEY` değerini **Secrets** bölümüne ekleyin.
- Tema ayarı `.streamlit/config.toml` ile gelir.
- Detaylı talimat: `DEPLOY_STREAMLIT_CLOUD.md`.