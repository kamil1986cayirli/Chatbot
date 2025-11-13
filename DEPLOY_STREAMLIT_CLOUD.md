# Deploy — Streamlit Cloud (v5.4, Azure destekli)

## 1) Repo
- Bu klasörü GitHub'a yükleyin.
- Dosyalar: `app.py`, `requirements.txt`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`

## 2) Streamlit Cloud
- New app → repo + branch → main file `app.py`

## 3) Secrets
- App → Settings → Secrets:
```
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-10-21"
```
(Gerekirse OPENAI_API_KEY de ekleyin.)

## 4) BO (opsiyonel)
- SAP BO PoC'yi `/bo` alt klasörü ile ekleyebiliriz (OpenDocument embed + REST export).

## Sorun giderme
- 401/403: Azure rol/yetki – anahtar ve endpoint'i kontrol edin.
- 429: Rate limit – düşük maliyet kip + dosya/adım sınırı.
- PDF upload: Azure Responses API destekler; file purpose `assistants` kullanın.