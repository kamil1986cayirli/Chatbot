# app.py (v5.3 Cloud Pack) — Görsel/PDF/CSV/Excel/Parquet analizi + Q&A + KPI + Sıralı analiz
import os, io, re, json, base64, time
from typing import Optional, Dict, Any, List

import streamlit as st
from PIL import Image
import requests
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="Dashboard & Rapor Analizörü — v5.3", page_icon="🧠", layout="wide")

# ---------- HELPERS ----------
def b64_from_image_bytes(img_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(img_bytes).decode("utf-8")

def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    brace = re.search(r"(\{[\s\S]*\})", text)
    if brace:
        try:
            return json.loads(brace.group(1))
        except Exception:
            return None
    return None

def build_instructions(detail_level: str, language: str, template_key: str, custom_template: str) -> str:
    base = f"""
    Sen üst düzey bir iş analisti ve veri görselleştirme uzmanısın.
    Görev: Kullanıcının yüklediği dashboard görseli, matbu rapor veya tablo dosyasını EN İNCE DETAYINA kadar incele ve aşağıdaki formatla yanıt ver.
    Yazım dili: {'Türkçe' if language == 'TR' else 'English'}.
    Ton: Kısa cümleler + net madde işaretleri, teknik ama anlaşılır.
    Derinlik: {detail_level}.

    Yanıt formatı (AYNI MESAJ İÇİNDE iki bölüm):
    1) **RAPOR**
       - Kısa özet
       - Veri yapısı ve metrikler
       - Dikkat çeken anomaliler/aykırılıklar
       - Trend/bağlam yorumları
       - Veri kalitesi/ölçek sorunları
       - Öneriler ve aksiyon maddeleri (etki/çaba tahmini ile)

    2) **JSON**
       ```json
       {{
         "summary": "string",
         "metrics": [{{"name": "string", "value": "string|number", "unit": "string|null"}}],
         "anomalies": [{{"title": "string", "where": "string", "why": "string"}}],
         "trends": [{{"signal": "up|down|flat|seasonal", "metric": "string", "confidence": "low|medium|high"}}],
         "quality_flags": [{{"issue": "string", "severity": "low|medium|high"}}],
         "recommendations": [{{"title": "string", "impact": "low|medium|high", "effort": "low|medium|high", "steps": ["..."]}}]
       }}
       ```

    Kurallar:
    - Dosyada/görselde/raporda olmayan sayı uydurma. Emin değilsen "belirsiz" de.
    - Birim ve tarihleri açık yaz.
    - Metin/tablolar okunabiliyorsa önemli alanları listele.
    """
    templates = {
        "Genel": "",
        "Satış Performansı": "- Gelir, brüt kâr, dönüşüm, AOV, iade oranı; kırılımlar ve kampanya etkisi.",
        "Pazarlama Kampanyası": "- Harcama, CTR, CPC, CPM, CPA, ROAS; kanal/yaratıcı karşılaştır.",
        "Finans Raporu": "- Gelir-gider, marjlar, nakit akışı, çalışma sermayesi metrikleri.",
        "IT Operasyonları": "- Olay sayısı, MTTR/MTBF, değişiklik başarısı, kapasite ve SLA.",
    }
    base = re.sub(r"\n[ \t]+", "\n", base).strip()
    extra = templates.get(template_key, "")
    custom = custom_template.strip() if custom_template else ""
    full = base + "\n\n" + extra + ("\n" + custom if custom else "")
    return full

def call_openai_on_image(client: "OpenAI", model: str, prompt: str, image_bytes: bytes, mime: str) -> str:
    data_url = b64_from_image_bytes(image_bytes, mime)
    response = client.responses.create(
        model=model,
        instructions=prompt,
        input=[
            {"role": "user", "content": [
                {"type": "input_text", "text": "Bu görseldeki dashboard/raporu ayrıntılı analiz et."},
                {"type": "input_image", "image_url": {"url": data_url}},
            ]}
        ],
    )
    return getattr(response, "output_text", None) or response.output[0].content[0].text

def call_openai_on_file(client: "OpenAI", model: str, prompt: str, file_name: str, file_bytes: bytes) -> str:
    # IMPORTANT: purpose="assistants" (Responses API ile uyumlu)
    uploaded = client.files.create(file=(file_name, io.BytesIO(file_bytes)), purpose="assistants")
    response = client.responses.create(
        model=model,
        instructions=prompt,
        input=[
            {"role": "user", "content": [
                {"type": "input_file", "file_id": uploaded.id},
                {"type": "input_text", "text": "Bu dosyadaki içeriği (PDF/rapor) ayrıntılı analiz et."},
            ]}
        ],
    )
    return getattr(response, "output_text", None) or response.output[0].content[0].text

def call_openai_on_table(client: "OpenAI", model: str, prompt: str, table_prompt: str) -> str:
    response = client.responses.create(
        model=model,
        instructions=prompt,
        input=[{"role": "user", "content": [{"type": "input_text", "text": table_prompt}]}],
    )
    return getattr(response, "output_text", None) or response.output[0].content[0].text

def call_openai_qa(client: "OpenAI", model: str, analysis_text: str, history: List[dict], user_question: str, lang: str) -> str:
    hist_text = ""
    for m in history[-10:]:
        role = "Kullanıcı" if m["role"] == "user" else "Asistan"
        hist_text += f"{role}: {m['text']}\n"
    qa_prompt = f"""
    Sen bir analiz danışmanısın. Aşağıdaki analiz çıktısını ve önceki mesajları bağlam al.
    Analiz (ham metin):
    ---
    {analysis_text}
    ---
    Önceki mesajlar:
    {hist_text}
    Yeni soru: {user_question}

    Cevap dili: {'Türkçe' if lang=='TR' else 'English'}.
    Yanıtın kısa, net ve mühendisçe olsun. Gerekirse maddeler kullan.
    """
    response = client.responses.create(
        model=model,
        instructions="Analizdeki bilgiye sadık kal. Belirsizse varsayım yapmadan 'emin değilim' de.",
        input=[{"role": "user", "content": [{"type": "input_text", "text": qa_prompt}]}],
    )
    return getattr(response, "output_text", None) or response.output[0].content[0].text

def fetch_from_url(url: str) -> Optional[tuple]:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        content_type = (r.headers.get("Content-Type") or "").lower()
        data = r.content
        u = url.lower()

        # PDF & Görseller
        if "pdf" in content_type or u.endswith(".pdf"):
            return ("application/pdf", data)
        if "png" in content_type or u.endswith(".png"):
            return ("image/png", data)
        if "jpeg" in content_type or "jpg" in content_type or u.endswith((".jpg",".jpeg")):
            return ("image/jpeg", data)

        # Tablo dosyaları
        if "text/csv" in content_type or u.endswith(".csv"):
            return ("text/csv", data)
        if "text/tab-separated-values" in content_type or u.endswith(".tsv"):
            return ("text/tsv", data)
        if "spreadsheet" in content_type or "sheet" in content_type or u.endswith((".xlsx",".xls")):
            return ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", data)
        if "parquet" in content_type or u.endswith(".parquet"):
            return ("application/parquet", data)

        return None
    except Exception:
        return None

def _limit_df(df: pd.DataFrame, max_rows: int = 100, max_cols: int = 50) -> pd.DataFrame:
    df2 = df.copy()
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    if df2.shape[0] > max_rows:
        df2 = df2.iloc[:max_rows, :]
    return df2

def dataframe_to_prompt(df: pd.DataFrame, file_name: str) -> str:
    df_limited = _limit_df(df)
    schema_lines = [f"- {c}: {str(df[c].dtype)}" for c in df_limited.columns]
    try:
        preview = df_limited.head(10).to_markdown(index=False)
    except Exception:
        preview = df_limited.head(10).to_csv(index=False)
    try:
        stats = df_limited.describe(include='all').transpose().fillna("").to_markdown()
    except Exception:
        stats = "(istatistik üretilemedi)"
    return (
        f"Dosya adı: {file_name}\n"
        f"Satır x Sütun: {df.shape[0]} x {df.shape[1]}\n\n"
        f"Şema:\n" + "\n".join(schema_lines) + "\n\n"
        f"İlk 10 satır:\n{preview}\n\n"
        f"Temel istatistikler (sınırlı):\n{stats}\n"
        f"\nYukarıdaki tablo özetini, verilen talimatlarla birlikte ayrıntılı analiz et."
    )

def read_table_file(file_name: str, file_bytes: bytes, mime: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    lname = file_name.lower()
    if mime == "text/csv" or lname.endswith(".csv"):
        return pd.read_csv(buf)
    if mime == "text/tsv" or lname.endswith(".tsv"):
        return pd.read_csv(buf, sep="\t")
    if "spreadsheetml" in mime or lname.endswith((".xlsx",".xls")):
        return pd.read_excel(buf)
    if "parquet" in mime or lname.endswith(".parquet"):
        return pd.read_parquet(buf)  # pyarrow gerekir
    # Son çare: csv dene
    return pd.read_csv(buf)

# ---------- UI ----------
st.title("🧠 Dashboard & Rapor Analizörü (v5.3)")
st.caption("Analiz + Q&A • Şablonlar • CSV/Excel/Parquet • KPI • Sıralı analiz + ilerleme çubuğu")

# Secrets/Env otomatik doldurma
default_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

st.sidebar.title("⚙️ Ayarlar")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=default_api_key)
model = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
detail = st.sidebar.selectbox("Detay seviyesi", ["çok yüksek", "yüksek", "orta"])
lang = st.sidebar.selectbox("Çıktı dili", ["TR", "EN"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Rapor Şablonu")
template_key = st.sidebar.selectbox("Şablon", ["Genel","Satış Performansı","Pazarlama Kampanyası","Finans Raporu","IT Operasyonları"], index=0)
custom_template = st.sidebar.text_area("Özel şablon ekle (opsiyonel)", height=120, placeholder="Ek talimatlar...")

st.sidebar.markdown("---")
kpi_txt = st.sidebar.text_area("KPI hedefleri (JSON veya 'Ad=Değer' satırları)", height=120, placeholder='{"Revenue": 1000000, "CR": 2.5}')
def parse_kpi_targets(txt: str) -> Dict[str, float]:
    if not txt: return {}
    txt = txt.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return {str(k): float(str(v).replace('%','').replace(',','').strip()) for k,v in obj.items()}
    except Exception:
        pass
    targets = {}
    for line in txt.splitlines():
        if "=" in line:
            k,v = line.split("=",1)
            k = k.strip()
            v = v.strip().replace("%","").replace(",","").replace(" ","")
            try:
                targets[k] = float(v)
            except Exception:
                continue
    return targets
kpi_targets = parse_kpi_targets(kpi_txt)

st.sidebar.markdown("---")
order_url_pos = st.sidebar.selectbox("URL dosyasının sırası", ["En sonda", "En başta"], index=0)
default_active_choice = st.sidebar.selectbox("Varsayılan aktif analiz", ["Son", "İlk"], index=0)

if not api_key:
    st.info("Sol menüden bir OpenAI API anahtarı girilmeli (veya Secrets'a eklenmeli).")
    st.stop()

client = OpenAI(api_key=api_key)

if "analyses" not in st.session_state:
    st.session_state["analyses"] = []  # list of dicts: {id, name, text, json}
if "chat" not in st.session_state:
    st.session_state["chat"] = {}      # analysis_id -> [{"role","text"}]

st.subheader("1) Dosya yükleyin veya URL verin")
colu, colv = st.columns([1,1])
with colu:
    uploaded_files = st.file_uploader(
        "Görsel (PNG/JPG) • PDF • Tablo (CSV/TSV/XLSX/XLS/Parquet)",
        type=["png","jpg","jpeg","pdf","csv","tsv","xlsx","xls","parquet"],
        accept_multiple_files=True
    )
with colv:
    url_input = st.text_input("Veya dosya URL'si", placeholder="https://... (.png/.jpg/.pdf/.csv/.xlsx/.parquet)")

user_notes = st.text_area("Notlar/Hedefler (opsiyonel)", height=100)

if st.button("Analizi Başlat", type="primary"):
    files_to_process = []

    # 1) Kullanıcının seçtiği sırayı koru
    if uploaded_files:
        for idx, f in enumerate(uploaded_files):
            files_to_process.append((idx, f.name, f.read(), f.type))

    # 2) URL dosyasını seçilen yere yerleştir
    if url_input:
        fetched = fetch_from_url(url_input.strip())
        if fetched:
            mime, data = fetched
            if order_url_pos == "En başta":
                files_to_process.insert(0, (-1, "from_url", data, mime))
            else:
                files_to_process.append((len(files_to_process), "from_url", data, mime))
        else:
            st.error("URL indirilemedi veya dosya tipi desteklenmiyor.")

    if not files_to_process:
        st.warning("Lütfen en az bir dosya seçin veya geçerli bir URL girin.")
    else:
        instructions = build_instructions(detail, lang, template_key, custom_template)
        if user_notes:
            instructions += f"\n\nKullanıcı notları/bağlam: {user_notes}\n"

        # 3) Sırayı kesinleştir
        files_to_process.sort(key=lambda x: x[0])

        total = len(files_to_process)
        progress = st.progress(0.0)

        for i, (order, file_name, file_bytes, mime) in enumerate(files_to_process, start=1):
            with st.spinner(f"{i}/{total} {file_name} analiz ediliyor..."):
                try:
                    # --- GÖRSEL ---
                    if mime in ("image/png", "image/jpeg"):
                        try:
                            img = Image.open(io.BytesIO(file_bytes))
                            st.image(img, caption=file_name, use_column_width=True)
                        except Exception:
                            pass
                        text = call_openai_on_image(client, model, instructions, file_bytes, mime)

                    # --- PDF ---
                    elif mime == "application/pdf":
                        text = call_openai_on_file(client, model, instructions, file_name if file_name!="from_url" else "from_url.pdf", file_bytes)

                    # --- TABLO DOSYALARI ---
                    elif mime in ("text/csv", "text/tsv",
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  "application/parquet") or file_name.lower().endswith((".csv",".tsv",".xlsx",".xls",".parquet")):
                        try:
                            df = read_table_file(file_name, file_bytes, mime)
                            st.write("Tablo önizleme (ilk 10 satır):")
                            st.dataframe(_limit_df(df, max_rows=10, max_cols=20), use_container_width=True)
                        except Exception as e:
                            st.error(f"{file_name}: Tablo okunamadı. Hata: {e}")
                            progress.progress(i/total)
                            continue
                        table_prompt = dataframe_to_prompt(df, file_name)
                        text = call_openai_on_table(client, model, instructions, table_prompt)

                    else:
                        st.error(f"{file_name}: Desteklenmeyen MIME tipi ({mime}).")
                        progress.progress(i/total)
                        continue

                    data = safe_json_extract(text)
                    analysis_id = str(int(time.time()*1000))
                    st.session_state["analyses"].append({
                        "id": analysis_id,
                        "name": file_name,
                        "text": text,
                        "json": data,
                    })
                    if "chat" not in st.session_state:
                        st.session_state["chat"] = {}
                    st.session_state["chat"][analysis_id] = []

                    st.success(f"{file_name} ✅ ( {i}/{total} )")
                except Exception as e:
                    st.exception(e)

            progress.progress(i/total)

# If we have analyses, let user pick one to review + chat
if st.session_state["analyses"]:
    st.subheader("2) Analizi görüntüleyin ve aynı pencerede soru sorun")
    options = [f"{a['name']} (id:{a['id']})" for a in st.session_state["analyses"]]
    default_idx = (len(options)-1) if default_active_choice=="Son" else 0
    chosen = st.selectbox("Aktif analiz", options, index=default_idx)
    active_id = st.session_state["analyses"][options.index(chosen)]["id"]
    active = next(a for a in st.session_state["analyses"] if a["id"] == active_id)

    st.markdown(f"### 🔎 {active['name']}")
    tabs = st.tabs(["Rapor", "JSON", "Grafik", "KPI", "Dışa Aktar", "Ham Çıktı"])
    report_only = active["text"].split("```json")[0].strip() if "```json" in active["text"] else active["text"]
    with tabs[0]:
        st.markdown(report_only)
    with tabs[1]:
        if active["json"] is not None:
            st.json(active["json"], expanded=False)
        else:
            st.warning("Geçerli JSON algılanamadı. Ham çıktıya bakın.")
    with tabs[2]:
        # JSON->Bar chart
        metrics = (active["json"] or {}).get("metrics") if active["json"] else []
        rows = []
        for m in metrics or []:
            name = m.get("name","")
            value = None
            try:
                value = float(str(m.get("value","")).replace("%","").replace(",","").strip())
            except Exception:
                value = None
            if name and value is not None:
                rows.append((name, value))
        if rows:
            names = [r[0] for r in rows][:30]
            vals  = [r[1] for r in rows][:30]
            fig = plt.figure()
            plt.bar(range(len(vals)), vals)
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.title("JSON -> Metrikler")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Sayısal metrik bulunamadı.")
    with tabs[3]:
        st.markdown("#### KPI Kartları")
        metrics = (active["json"] or {}).get("metrics") if active["json"] else []
        parsed = []
        for m in metrics or []:
            name = m.get("name","")
            val = None
            try:
                val = float(str(m.get("value","")).replace("%","").replace(",","").strip())
            except Exception:
                val = None
            unit = m.get("unit")
            if (val is not None) and name:
                parsed.append((name, val, unit))
        if parsed:
            parsed = parsed[:6]
            cols = st.columns(len(parsed))
            for i,(name,val,unit) in enumerate(parsed):
                target = (kpi_targets or {}).get(name)
                if target is not None:
                    delta = val - target
                    delta_str = f"{delta:.2f}" if unit is None else f"{delta:.2f} {unit}"
                    cols[i].metric(name, f"{val:.2f}" + (f" {unit}" if unit else ""), delta=delta_str)
                else:
                    cols[i].metric(name, f"{val:.2f}" + (f" {unit}" if unit else ""))
        else:
            st.info("Sayısal KPI bulunamadı.")
    with tabs[4]:
        if active["json"]:
            j = active["json"]
            def df_from_list_of_dicts(rows, columns):
                if not rows: return None
                df = pd.DataFrame(rows)
                avail = [c for c in columns if c in df.columns]
                if avail: df = df[avail]
                return df
            def csv_bytes_from_df(df): return df.to_csv(index=False).encode("utf-8")
            sections = {
                "metrics": ["name","value","unit"],
                "anomalies": ["title","where","why"],
                "trends": ["signal","metric","confidence"],
                "quality_flags": ["issue","severity"],
                "recommendations": ["title","impact","effort","steps"],
            }
            for sec, cols in sections.items():
                rows = j.get(sec) or []
                df = df_from_list_of_dicts(rows, cols)
                if df is not None:
                    st.write(f"**{sec}** ({len(df)} kayıt)")
                    st.dataframe(df, use_container_width=True)
                    st.download_button(f"⬇️ {sec}.csv indir", data=csv_bytes_from_df(df), file_name=f"{sec}.csv", mime="text/csv")
                else:
                    st.write(f"**{sec}**: veri yok")
            st.download_button("⬇️ Rapor (Markdown) indir", data=active["text"], file_name=f"{active['name']}_analysis.md", mime="text/markdown")
        else:
            st.info("JSON verisi yok, dışa aktarım yapılamıyor.")
    with tabs[5]:
        st.code(active["text"])

    st.markdown("---")
    st.markdown("### 💬 Rapor Hakkında Takip Soruları")
    for m in st.session_state["chat"][active_id]:
        with st.chat_message("assistant" if m["role"]=="assistant" else "user"):
            st.markdown(m["text"])

    user_q = st.chat_input("Bu rapor hakkında sorunuzu yazın…")
    if user_q:
        st.session_state["chat"][active_id].append({"role":"user","text":user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Yanıt hazırlanıyor..."):
                try:
                    ans = call_openai_qa(OpenAI(api_key=api_key), model, active["text"], st.session_state["chat"][active_id], user_q, lang)
                except Exception as e:
                    ans = f"Hata: {e}"
                st.markdown(ans)
                st.session_state["chat"][active_id].append({"role":"assistant","text":ans})

else:
    st.info("Önce bir analiz oluşturun.")

st.markdown("---")
st.caption("v5.3 — Cloud Pack: Streamlit Secrets uyumluluğu + PDF fix + CSV/TSV/XLSX/XLS/Parquet desteği.")