# Dashboard & Rapor Analizörü — v5.4 (Azure destekli)

- **Sağlayıcı seçimi**: OpenAI veya Azure OpenAI
- **Formatlar**: PNG/JPG, PDF, CSV/TSV, XLSX/XLS, Parquet
- **Çıktılar**: İnsan okunur rapor + JSON + KPI + grafik + CSV/Markdown dışa aktarım
- **Q&A**: Aynı pencerede takip soruları
- **Sırayla analiz + ilerleme çubuğu**
- **RateLimit**: Retry/backoff + "Düşük maliyet modu"

## Lokal çalıştırma
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-10-21"
streamlit run app.py
```

## Streamlit Cloud
- Secrets'a Azure anahtarların ekleyin.
- Uygulamayı açıp **Sağlayıcı: Azure OpenAI** seçin, deployment adınızı girin (ör. `gpt-4o`).