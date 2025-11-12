# 🧠 Dashboard & Matbu Rapor Analizörü (Streamlit + OpenAI)

Görsel dashboard ekran görüntüleri ve/veya **matbu rapor (PDF)** dosyalarını **OpenAI Responses API** ile ayrıntılı yorumlayan bir Streamlit uygulaması.

## Özellikler
- PNG/JPG **görseller** ve **PDF** dosyalarını kabul eder
- **gpt-4o** ve **gpt-4o-mini** ile çok ayrıntılı analiz
- Hem **insan okunur rapor** hem de **makine okunur JSON** döndürür
- Çıktıları JSON/Markdown olarak indirebilirsiniz

## Hızlı Başlangıç (macOS Terminal)
```bash
# 1) Proje klasörü
cd ~/Desktop
cp -R /mnt/data/gpt-dashboard-analyzer ./gpt-dashboard-analyzer
cd gpt-dashboard-analyzer

# 2) Sanal ortam (opsiyonel ama önerilir)
python3 -m venv .venv
source .venv/bin/activate

# 3) Paketler
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) API anahtarı
export OPENAI_API_KEY="sk-..."
# ya da .env kullanıyorsanız: cp .env.example .env ve içerisine anahtarı yazın

# 5) Uygulamayı başlat
streamlit run app.py
```

Uygulama açıldıktan sonra sol menüden modeli ve seçenekleri belirleyip görsel/PDF yükleyin.

## Sorular
- **Hangi modeller?** Varsayılan: `gpt-4o`, alternatif `gpt-4o-mini`. Her ikisi de metin+görsel girişi destekler.
- **PDF nasıl işleniyor?** Dosya, OpenAI API'ye `input_file` olarak yüklenir ve model tarafından çözümlenir.
- **Görseller nasıl iletiliyor?** Görsel, `data:` URL (base64) olarak `input_image` ile gönderilir.
- **JSON neden bazen başarısız?** Model çıktısı her zaman mükemmel JSON olmayabilir; uygulama önce ```json çitini, sonra gevşek bir { ... } bloğunu parse etmeye çalışır.

## Güvenlik/Veri
- Dosyalar model analizi için OpenAI'ye gönderilir.
- Gizli veriler içeriyorsa kurum politikalarınıza uygun hareket edin.

## Sorun Giderme
- `ModuleNotFoundError: openai` → `pip install -r requirements.txt`
- `OpenAIAuthenticationError` → API anahtarını girin veya `export OPENAI_API_KEY=...`
- `streamlit not found` → `pip install streamlit`

# v2 — Paylaşılabilir Link + Yorum/QA + URL'den Dosya

Bu sürümde:
- **Paylaşılabilir link**: Her analiz için benzersiz `?id=...` oluşturulur. Bu ID ile sayfa herkese açık paylaşılabilir.
- **Yorum & Soru-Cevap**: Analiz sayfasında herkes yorum bırakabilir, ek sorular sorabilir; model bağlamı kullanarak yanıtlar.
- **URL ile içeri aktarma**: PNG/JPG/PDF adresini girerek dosyayı uzaktan çekebilirsiniz.

## Dağıtım (Streamlit Community Cloud — önerilen)
1) Bu klasörü bir GitHub repo'su olarak push edin.
2) https://share.streamlit.io üzerinden repo'yu seçip `app.py` ile deploy edin.
3) Açılan genel URL'nin sonuna `?id=ANALIZ_ID` ekleyerek belirli bir analizi paylaşın.
4) **Önemli**: Genel kullanımda ziyaretçilerin kendi **OpenAI API Key**'lerini girmesini tercih edin. Aksi halde tüm kullanım maliyeti sizin anahtarınıza yazılır.

## Güvenlik Notları
- Bu app, ziyaretçinin girdiği API anahtarıyla çalışabilir. Sunucu tarafında ortam değişkeni tanımlarsanız, tüm çağrılar sizin anahtarınızdan geçer (maliyet riski).
- Hassas veriler paylaşılmadan önce maskeleme yapın.
- Çok yoğun kullanımda SQLite yerine (Supabase/Postgres/Firestore) kullanmanız tavsiye edilir.

## v3 — Eklenen 4 Özellik
1. **Kimlik Doğrulama (Admin Parola)**  
   - `ADMIN_PASS` ortam değişkeni ile admin girişi.
   - Admin yoksa “demo modu” uyarısı.
2. **Ortak/Davet/Private Odalar (ACL)**  
   - Her analiz için `public | unlisted | private` görünürlük.
   - Private için erişim kodu veya **davet token** ile giriş.
3. **Rate Limit + Kötüye Kullanım Koruması**  
   - `RATE_ANALYZE_PER_10MIN` ve `RATE_ASK_PER_10MIN` ile sınırlar.
   - Basit uygunsuz dil/uzunluk filtresi.
4. **Otomatik Grafik/Tablo (JSON → Matplotlib)**  
   - `metrics` listesindeki sayısal değerlerden çubuk grafik.
   - Grafik “Analiz” ve “Yorum” sekmelerinde gösterilebilir.

### Ortam Değişkenleri
```bash
export OPENAI_API_KEY="sk-..."
export ADMIN_PASS="parolaniz"
export ALLOW_PUBLIC_ANALYZE="false"          # true yaparsanız herkes analiz başlatır
export ROOM_DEFAULT_VISIBILITY="unlisted"     # public | unlisted | private
export RATE_ANALYZE_PER_10MIN="5"
export RATE_ASK_PER_10MIN="10"
```

## v4 — Tek Sayfa: Analiz + Aynı Pencerede Soru-Cevap
Bu sürüm tam olarak şu ihtiyacı hedefler:
- Raporu/görseli yükle, analiz et.
- **Aynı pencerede** (ayrı sekmeye geçmeden, ID paylaşmadan) rapora dair **takip sorularını** sor ve cevap al.
- Uygulamayı **Streamlit Cloud**'a deploy edip **tek bir genel link** paylaşmanız yeterli; linke sahip herkes girebilir.

### Dağıtım (Streamlit Cloud)
1) Bu klasörü bir GitHub repo'suna push edin.  
2) Streamlit Community Cloud → repo → `app.py` ile deploy.  
3) Açılan URL'yi paylaşın. Kimlik doğrulama/ID gerekmez.

> Not: Çalışma maliyetini kontrol etmek için ziyaretçilerden **kendi OpenAI API Key**'lerini sol menüden girmelerini isteyebilirsiniz. Ya da sunucu tarafında `OPENAI_API_KEY` tanımlayıp tek anahtar da kullanabilirsiniz.

## v5 — Şablonlar + CSV Dışa Aktarım + KPI Kartları
- **Rapor Şablonları**: Genel, Satış, Pazarlama, Finans, IT — ayrıca özel şablon metni ekleyebilirsiniz.
- **CSV Dışa Aktarım**: JSON içindeki `metrics / anomalies / trends / quality_flags / recommendations` bölümlerini tablo olarak görüp CSV indirebilirsiniz.
- **KPI Kartları**: Sidebar'a hedefleri JSON veya `Ad=Değer` şeklinde yazın; metriklerle karşılaştırmalı KPI kartları oluşur.

> Not: Grafikler matplotlib ile üretilir (tek grafik / renk belirtilmeden). Ziyaretçiler kendi API anahtarını girebilir.

## v5.1 — Sıralı Analiz + İlerleme Çubuğu
- Yüklediğiniz dosyalar **tam seçtiğiniz sırayla** işlenir.
- Üstte **ilerleme çubuğu** ve her dosya için `i/total` durum mesajı vardır.
- Sidebar’dan **URL dosyasının sırası** (en başta / en sonda) ve **varsayılan aktif analiz** (son / ilk) seçilebilir.