# Deploy — Streamlit Cloud (v5.3 Cloud Pack)

## 1) Repo hazırlığı
- Bu klasörü bir Git repo olarak yayınlayın (GitHub, GitLab).
- `app.py`, `requirements.txt`, `.streamlit/config.toml` dosyaları yeterli.

## 2) Streamlit Cloud
- https://streamlit.io/cloud > **New app** > repoyu seçin.
- **Main file path**: `app.py`

## 3) Secrets
- App > **Settings** > **Secrets** bölümüne aşağıdakini girin:
```
OPENAI_API_KEY="sk-..."
```
(İsterseniz Azure OpenAI için endpoint/key de ekleyin.)

## 4) Çalıştırma
- Deploy tamamlanınca linki paylaşın.
- Kullanıcılar sol menüde anahtar girmek zorunda kalmadan (Secrets ile) hazır çalışır.

## Sorun giderme
- Python bağımlılık: `requirements.txt`
- Bellek hatası: Çok büyük dosyalarda tablo özetleyici limiti artırmayın (varsayılan 100 satır/50 sütun).
- OpenAI hatası: Key yanlış/limit dolmuş olabilir; loglarda HTTP 401/429 görünür.