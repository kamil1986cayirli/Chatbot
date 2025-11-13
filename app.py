# app.py — Dashboard & Rapor Analizörü v6.1.2
# OpenAI + Azure OpenAI + Google Gemini • UI: 3 sekme • 429 backoff + mini fallback

import io
import re
import os
import json
import time
import base64
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st
from PIL import Image
import requests
import matplotlib.pyplot as plt
import pandas as pd

# -------------------- SDK'lar --------------------
try:
    from openai import OpenAI, AzureOpenAI
except Exception:
    OpenAI = AzureOpenAI = None

try:
    import google.generativeai as genai
    from google.api_core.exceptions import NotFound
except Exception:
    genai = None

    class NotFound(Exception):
        pass


st.set_page_config(page_title="Dashboard & Rapor Analizörü", page_icon="🧠", layout="wide")

# -------------------- Stil --------------------
CUSTOM_CSS = """
<style>
section.main > div { padding-top: 1rem; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

.card {
  border-radius: 14px;
  border: 1px solid #e5e7eb;
  padding: 14px 16px;
  background: #ffffff;
  box-shadow: 0 2px 10px rgba(2,6,23,.04);
}
.card h4 { margin: 0 0 6px 0; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: #eef2ff;
  color: #3730a3;
  font-size: 12px;
  margin-left: 8px;
}
hr.soft { border: none; border-top: 1px solid #eef2f7; margin: 18px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Yardımcılar --------------------
def b64_from_bytes(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\{[\s\S]*\})", text)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            return None
    return None


def _limit_df(df: pd.DataFrame, max_rows: int = 100, max_cols: int = 50) -> pd.DataFrame:
    df2 = df.copy()
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    if df2.shape[0] > max_rows:
        df2 = df2.iloc[:max_rows, :]
    return df2


def dataframe_to_prompt(df: pd.DataFrame, file_name: str, low_cost: bool = False) -> str:
    df_limited = df.head(50) if low_cost else _limit_df(df)
    schema_lines = [f"- {c}: {str(df[c].dtype)}" for c in df_limited.columns[: (10 if low_cost else 50)]]
    try:
        preview = df_limited.head(5 if low_cost else 10).to_markdown(index=False)
    except Exception:
        preview = df_limited.head(5 if low_cost else 10).to_csv(index=False)
    if low_cost:
        stats = "(istatistik hesaplanmadı — düşük maliyet modu)"
    else:
        try:
            stats = df_limited.describe(include="all").transpose().fillna("").to_markdown()
        except Exception:
            stats = "(istatistik üretilemedi)"
    return (
        f"Dosya adı: {file_name}\n"
        f"Satır x Sütun: {df.shape[0]} x {df.shape[1]}\n\n"
        f"Şema:\n" + "\n".join(schema_lines) + "\n\n"
        f"İlk satırlar:\n{preview}\n\n"
        f"Temel istatistikler:\n{stats}\n"
        f"\nYukarıdaki tablo özetini, verilen talimatlarla birlikte {'kısa ve öz' if low_cost else 'ayrıntılı'} analiz et."
    )


def read_table_file(file_name: str, file_bytes: bytes, mime: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    lname = file_name.lower()
    if mime == "text/csv" or lname.endswith(".csv"):
        return pd.read_csv(buf)
    if mime == "text/tsv" or lname.endswith(".tsv"):
        return pd.read_csv(buf, sep="\t")
    if "spreadsheetml" in mime or lname.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    if "parquet" in mime or lname.endswith(".parquet"):
        return pd.read_parquet(buf)
    return pd.read_csv(buf)


def fetch_from_url(url: str) -> Optional[tuple]:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "").lower()
        data = r.content
        u = url.lower()
        if "pdf" in ct or u.endswith(".pdf"):
            return ("application/pdf", data)
        if "png" in ct or u.endswith(".png"):
            return ("image/png", data)
        if "jpeg" in ct or "jpg" in ct or u.endswith((".jpg", ".jpeg")):
            return ("image/jpeg", data)
        if "text/csv" in ct or u.endswith(".csv"):
            return ("text/csv", data)
        if "text/tab-separated-values" in ct or u.endswith(".tsv"):
            return ("text/tsv", data)
        if "spreadsheet" in ct or "sheet" in ct or u.endswith((".xlsx", ".xls")):
            return ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", data)
        if "parquet" in ct or u.endswith(".parquet"):
            return ("application/parquet", data)
        return None
    except Exception:
        return None


def build_instructions(detail_level: str, language: str, template_key: str, custom_template: str, low_cost: bool = False) -> str:
    base = f"""
    Sen üst düzey bir iş analisti ve veri görselleştirme uzmanısın.
    Görev: Kullanıcının yüklediği dashboard görseli, matbu rapor veya tablo dosyasını EN İNCE DETAYINA kadar incele.
    Yazım dili: {'Türkçe' if language == 'TR' else 'English'}. Derinlik: {detail_level}.
    Yanıt formatı:
    1) RAPOR — özet, metrikler, anomaliler, trendler, veri kalitesi, öneriler.
    2) JSON — geçerli JSON üret (summary, metrics, anomalies, trends, quality_flags, recommendations).
    Kurallar: Uydurma sayı yok; birimleri açık yaz; belirsizse belirt.
    """
    templates = {
        "Genel": "",
        "Satış Performansı": "- Gelir, brüt kâr, dönüşüm, AOV, iade oranı; kanal/bölge/ürün kırılımları; kampanya etkisi.",
        "Pazarlama Kampanyası": "- Harcama, gösterim, tıklama, CTR, CPC, CPA, ROAS; kanal/yaratıcı/segment karşılaştır.",
        "Finans Raporu": "- Gelir-gider, marjlar, nakit akışı, çalışma sermayesi metrikleri, dönem karşılaştırmaları.",
        "IT Operasyonları": "- Incident/MTTR/MTBF, SLA ihlalleri, kapasite ve kök neden örüntüleri.",
    }
    base = re.sub(r"\n[ \t]+", "\n", base).strip()
    extra = templates.get(template_key, "")
    custom = custom_template.strip() if custom_template else ""
    if low_cost:
        base += "\n\nKısıt: Token tasarrufu yap; yalnızca en kritik bulguları yaz."
    return base + ("\n\n" + extra if extra else "") + ("\n" + custom if custom else "")


# -------------------- Sağlayıcılar --------------------
class Provider:
    OPENAI = "OpenAI"
    AZURE = "Azure OpenAI"
    GEMINI = "Google Gemini"


def make_clients(provider: str, creds: dict):
    if provider == Provider.OPENAI:
        if OpenAI is None:
            raise RuntimeError("openai paketi kurulu değil.")
        client = OpenAI(api_key=creds["OPENAI_API_KEY"])
        return {"client": client, "model": creds["model"]}
    elif provider == Provider.AZURE:
        if AzureOpenAI is None:
            raise RuntimeError("openai paketi kurulu değil.")
        client = AzureOpenAI(
            api_key=creds["AZURE_OPENAI_API_KEY"],
            azure_endpoint=creds["AZURE_OPENAI_ENDPOINT"],
            api_version=creds.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        return {"client": client, "model": creds["deployment"]}
    else:
        if genai is None:
            raise RuntimeError("google-generativeai paketi kurulu değil.")
        genai.configure(api_key=creds["GEMINI_API_KEY"])
        model_name = creds["gemini_model"]
        model = genai.GenerativeModel(model_name)
        return {"client": model, "model": model_name}


# -------------------- Backoff yardımcıları --------------------
def _with_backoff(fn, tries: int = 3, sleep_base: float = 1.5):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e).lower()
            if any(k in msg for k in ("429", "rate limit", "insufficient_quota")) and i < tries - 1:
                time.sleep(sleep_base * (i + 1))
                continue
            raise
    raise last


# -------------------- Gemini safe call --------------------
def gemini_generate_content(model, parts):
    try:
        return model.generate_content(parts)
    except NotFound as e:
        name = getattr(model, "model_name", None) or getattr(model, "_model", None)
        try_name = f"models/{name}" if name and not str(name).startswith("models/") else name
        if try_name:
            try:
                m2 = type(model)(try_name)
                return m2.generate_content(parts)
            except Exception:
                raise e
        raise e


# -------------------- Model çağrıları --------------------
def call_on_image(provider: str, client_obj, model_name: str, prompt: str, image_bytes: bytes, mime: str) -> str:
    if provider in (Provider.OPENAI, Provider.AZURE):
        def _req(md):
            return client_obj.responses.create(
                model=md,
                instructions=prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Bu görseldeki dashboard/raporu ayrıntılı analiz et."},
                            {"type": "input_image", "image_url": {"url": b64_from_bytes(image_bytes, mime)}},
                        ],
                    }
                ],
            )

        try:
            resp = _with_backoff(lambda: _req(model_name))
        except Exception:
            resp = _req("gpt-4o-mini")
        return getattr(resp, "output_text", None) or resp.output[0].content[0].text
    else:
        image_part = {"mime_type": mime, "data": image_bytes}
        resp = gemini_generate_content(client_obj, [prompt, image_part])
        return resp.text


def call_on_pdf(provider: str, client_obj, model_name: str, prompt: str, file_name: str, file_bytes: bytes) -> str:
    if provider in (Provider.OPENAI, Provider.AZURE):
        uploaded = client_obj.files.create(file=(file_name, io.BytesIO(file_bytes)), purpose="assistants")

        def _req(md):
            return client_obj.responses.create(
                model=md,
                instructions=prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded.id},
                            {"type": "input_text", "text": "Bu dosyadaki (PDF/rapor) içeriği analiz et."},
                        ],
                    }
                ],
            )

        try:
            resp = _with_backoff(lambda: _req(model_name))
        except Exception:
            resp = _req("gpt-4o-mini")
        return getattr(resp, "output_text", None) or resp.output[0].content[0].text
    else:
        uploaded = genai.upload_file(bytes=file_bytes, mime_type="application/pdf", display_name=file_name)
        resp = gemini_generate_content(client_obj, [prompt, uploaded])
        return resp.text


def call_on_table(provider: str, client_obj, model_name: str, prompt: str, table_prompt: str) -> str:
    if provider in (Provider.OPENAI, Provider.AZURE):
        def _req(md, text):
            return client_obj.responses.create(
                model=md,
                instructions=prompt,
                input=[{"role": "user", "content": [{"type": "input_text", "text": text}]}],
            )

        try:
            resp = _with_backoff(lambda: _req(model_name, table_prompt))
        except Exception:
            short = table_prompt[:2000]
            resp = _req("gpt-4o-mini", short)
        return getattr(resp, "output_text", None) or resp.output[0].content[0].text
    else:
        resp = gemini_generate_content(client_obj, [prompt, table_prompt])
        return resp.text


def call_for_qa(provider: str, client_obj, model_name: str, analysis_text: str, history: List[dict], user_question: str, lang: str) -> str:
    hist = ""
    for m in history[-10:]:
        role = "Kullanıcı" if m["role"] == "user" else "Asistan"
        hist += f"{role}: {m['text']}\n"

    qa_prompt = f"""
    Sen bir analiz danışmanısın. Aşağıdaki analiz çıktısını ve önceki mesajları bağlam al.
    Analiz:
    ---
    {analysis_text}
    ---
    Önceki mesajlar:
    {hist}
    Yeni soru: {user_question}
    Cevap dili: {'Türkçe' if lang=='TR' else 'English'}.
    Kısa ve net yanıt ver.
    """

    if provider in (Provider.OPENAI, Provider.AZURE):
        def _req(md, text):
            return client_obj.responses.create(
                model=md,
                instructions="Analizdeki bilgiye sadık kal. Belirsizse varsayım yapma.",
                input=[{"role": "user", "content": [{"type": "input_text", "text": text}]}],
            )

        try:
            resp = _with_backoff(lambda: _req(model_name, qa_prompt))
        except Exception:
            resp = _req("gpt-4o-mini", qa_prompt[:2000])
        return getattr(resp, "output_text", None) or resp.output[0].content[0].text
    else:
        resp = gemini_generate_content(client_obj, qa_prompt)
        return resp.text


# -------------------- UI --------------------
st.markdown("## 🧠 Dashboard & Rapor Analizörü")
st.caption("Derli toplu arayüz: 1) Bağlantı → 2) Yükle & Çalıştır → 3) Sonuçlar & Q&A")
tabs = st.tabs(["🔌 Bağlantı", "📤 Yükle & Çalıştır", "📊 Sonuçlar & Q&A"])

with tabs[0]:
    st.subheader("Sağlayıcı ve Kimlik Bilgileri")
    colA, colB = st.columns([1, 1])
    with colA:
        provider = st.radio("Sağlayıcı", ["OpenAI", "Azure OpenAI", "Google Gemini"], index=1, horizontal=True)
    with colB:
        low_cost = st.toggle("🔋 Düşük maliyet modu", value=True)

    if provider == "OpenAI":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        openai_key = st.text_input("OPENAI_API_KEY", type="password", value="")
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=1)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Hazır mı?"):
            st.session_state["conn"] = {"provider": provider, "OPENAI_API_KEY": openai_key, "model": model}
            st.success("Kaydedildi.")

    elif provider == "Azure OpenAI":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        az_key = st.text_input("AZURE_OPENAI_API_KEY", type="password", value="")
        az_ep = st.text_input("AZURE_OPENAI_ENDPOINT", value="", placeholder="https://<resource>.openai.azure.com/")
        az_ver = st.text_input("AZURE_OPENAI_API_VERSION", value="2024-10-21")
        deployment = st.text_input("Deployment name", value="gpt-4o-mini")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Hazır mı?"):
            st.session_state["conn"] = {
                "provider": provider,
                "AZURE_OPENAI_API_KEY": az_key,
                "AZURE_OPENAI_ENDPOINT": az_ep,
                "AZURE_OPENAI_API_VERSION": az_ver,
                "deployment": deployment,
            }
            st.success("Kaydedildi.")

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        gemini_key = st.text_input("GEMINI (GOOGLE_API_KEY)", type="password", value="")
        gemini_model = st.selectbox("Gemini modeli", ["gemini-1.5-flash-001", "gemini-1.5-pro-001"], index=0)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Hazır mı?"):
            st.session_state["conn"] = {
                "provider": provider,
                "GEMINI_API_KEY": gemini_key,
                "gemini_model": gemini_model,
            }
            st.success("Kaydedildi.")

    st.info("Devam etmeden önce yukarıda **Hazır mı?** butonuna basarak bağlantı bilgilerini kaydedin.")

with tabs[1]:
    st.subheader("Yükleme ve Analiz")
    if "conn" not in st.session_state:
        st.warning("Önce 'Bağlantı' sekmesinde bilgileri kaydedin.")
        st.stop()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        detail = st.selectbox("Detay seviyesi", ["çok yüksek", "yüksek", "orta"], index=1)
    with c2:
        lang = st.selectbox("Dil", ["TR", "EN"], index=0)
    with c3:
        template_key = st.selectbox("Şablon", ["Genel", "Satış Performansı", "Pazarlama Kampanyası", "Finans Raporu", "IT Operasyonları"], index=0)
    with c4:
        max_files = st.slider("Maks. dosya", 1, 10, 3)

    colU, colR = st.columns([1, 1])
    with colU:
        uploaded_files = st.file_uploader(
            "Dosya yükleyin (PNG/JPG, PDF, CSV/TSV, XLSX/XLS, Parquet)",
            type=["png", "jpg", "jpeg", "pdf", "csv", "tsv", "xlsx", "xls", "parquet"],
            accept_multiple_files=True,
        )
    with colR:
        url_input = st.text_input("Veya dosya URL'si", placeholder="https://... (.png/.jpg/.pdf/.csv/.xlsx/.parquet)")
        order_url_pos = st.selectbox("URL dosyasının sırası", ["En sonda", "En başta"], index=0)

    custom_template = st.text_area("Özel talimatlar (opsiyonel)", height=100, placeholder="Ek bağlam/amaç/KPI vb.")

    if "analyses" not in st.session_state:
        st.session_state["analyses"] = []
    if "chat" not in st.session_state:
        st.session_state["chat"] = {}

    if st.button("🚀 Analizi Başlat", type="primary"):
        conn = st.session_state["conn"]
        prov = conn["provider"]

        if prov == "OpenAI":
            client_obj = OpenAI(api_key=conn["OPENAI_API_KEY"])
            model_name = conn["model"]
        elif prov == "Azure OpenAI":
            client_obj = AzureOpenAI(
                api_key=conn["AZURE_OPENAI_API_KEY"],
                azure_endpoint=conn["AZURE_OPENAI_ENDPOINT"],
                api_version=conn["AZURE_OPENAI_API_VERSION"],
            )
            model_name = conn["deployment"]
        else:
            genai.configure(api_key=conn["GEMINI_API_KEY"])
            client_obj = genai.GenerativeModel(conn["gemini_model"])
            model_name = conn["gemini_model"]

        files_to_process: List[Tuple[int, str, bytes, str]] = []
        if uploaded_files:
            for idx, f in enumerate(uploaded_files):
                files_to_process.append((idx, f.name, f.read(), f.type))

        if url_input:
            fetched = fetch_from_url(url_input.strip())
            if fetched:
                mime, data = fetched
                files_to_process.insert(0 if order_url_pos == "En başta" else len(files_to_process), (-1, "from_url", data, mime))
            else:
                st.error("URL indirilemedi veya dosya tipi desteklenmiyor.")

        if not files_to_process:
            st.warning("Lütfen en az bir dosya yükleyin veya geçerli bir URL girin.")
        else:
            if len(files_to_process) > max_files:
                st.info(f"{len(files_to_process)} dosya seçildi, limit {max_files}. İlk {max_files} dosya analiz edilecek.")
                files_to_process = files_to_process[:max_files]

            if st.session_state.get("low_cost", True) and detail == "çok yüksek":
                detail = "orta"
            instructions = build_instructions(detail, lang, template_key, custom_template, low_cost=st.session_state.get("low_cost", True))

            files_to_process.sort(key=lambda x: x[0])
            total = len(files_to_process)
            with st.status("Analiz başlatıldı…", expanded=True) as status:
                for i, (order, file_name, file_bytes, mime) in enumerate(files_to_process, start=1):
                    st.write(f"**{i}/{total}** · {file_name} işleniyor…")
                    try:
                        if mime in ("image/png", "image/jpeg"):
                            try:
                                st.image(Image.open(io.BytesIO(file_bytes)), caption=file_name, use_column_width=True)
                            except Exception:
                                pass
                            text = call_on_image(prov, client_obj, model_name, instructions, file_bytes, mime)

                        elif mime == "application/pdf":
                            text = call_on_pdf(prov, client_obj, model_name, instructions, file_name if file_name != "from_url" else "from_url.pdf", file_bytes)

                        elif (
                            mime in ("text/csv", "text/tsv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/parquet")
                            or file_name.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".parquet"))
                        ):
                            try:
                                df = read_table_file(file_name, file_bytes, mime)
                                st.dataframe(_limit_df(df, 10, 20), use_container_width=True)
                            except Exception as e:
                                st.error(f"{file_name}: Tablo okunamadı. Hata: {e}")
                                continue
                            table_prompt = dataframe_to_prompt(df, file_name, low_cost=True)
                            text = call_on_table(prov, client_obj, model_name, instructions, table_prompt)

                        else:
                            st.error(f"{file_name}: Desteklenmeyen MIME tipi ({mime}).")
                            continue

                        data = safe_json_extract(text)
                        analysis_id = str(int(time.time() * 1000))
                        st.session_state["analyses"].append({"id": analysis_id, "name": file_name, "text": text, "json": data})
                        st.session_state["chat"][analysis_id] = []
                        st.success(f"{file_name} tamamlandı.")
                    except Exception as e:
                        st.exception(e)
                status.update(label="Analiz tamamlandı", state="complete")

with tabs[2]:
    st.subheader("Sonuçlar & Takip Soruları")
    if "analyses" not in st.session_state or not st.session_state["analyses"]:
        st.info("Henüz analiz yok. 'Yükle & Çalıştır' sekmesinden başlayın.")
        st.stop()

    options = [f"{a['name']} (id:{a['id']})" for a in st.session_state["analyses"]]
    chosen = st.selectbox("Aktif analiz", options, index=len(options) - 1)
    active_id = st.session_state["analyses"][options.index(chosen)]["id"]
    active = next(a for a in st.session_state["analyses"] if a["id"] == active_id)

    colL, colR = st.columns([2, 1], gap="large")
    with colL:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📄 Rapor")
        report_only = active["text"].split("```json")[0].strip() if "```json" in active["text"] else active["text"]
        st.markdown(report_only)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 💬 Takip Soruları")
        for m in st.session_state["chat"][active_id]:
            with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
                st.markdown(m["text"])
        user_q = st.chat_input("Bu rapor hakkında sorunuzu yazın…")
        if user_q:
            st.session_state["chat"][active_id].append({"role": "user", "text": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            conn = st.session_state.get("conn", {})
            prov = conn.get("provider")
            try:
                if prov == "OpenAI":
                    client_obj = OpenAI(api_key=conn["OPENAI_API_KEY"])
                    model_name = conn["model"]
                elif prov == "Azure OpenAI":
                    client_obj = AzureOpenAI(
                        api_key=conn["AZURE_OPENAI_API_KEY"],
                        azure_endpoint=conn["AZURE_OPENAI_ENDPOINT"],
                        api_version=conn["AZURE_OPENAI_API_VERSION"],
                    )
                    model_name = conn["deployment"]
                else:
                    genai.configure(api_key=conn["GEMINI_API_KEY"])
                    client_obj = genai.GenerativeModel(conn["gemini_model"])
                    model_name = conn["gemini_model"]

                # Q&A çağrısı
                hist = st.session_state["chat"][active_id]
                ans = call_for_qa(prov, client_obj, model_name, active["text"], hist, user_q, "TR")
            except Exception as e:
                ans = f"Hata: {e}"

            with st.chat_message("assistant"):
                st.markdown(ans)
            st.session_state["chat"][active_id].append({"role": "assistant", "text": ans})
        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('#### 🧾 JSON <span class="badge">makine-okur</span>', unsafe_allow_html=True)
        if active["json"] is not None:
            st.json(active["json"], expanded=False)
        else:
            st.warning("Geçerli JSON algılanamadı.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 KPI / Grafik")
        j = active["json"] or {}
        metrics = j.get("metrics") or []
        rows = []
        for m in metrics:
            name = m.get("name", "")
            try:
                value = float(str(m.get("value", "")).replace("%", "").replace(",", "").strip())
            except Exception:
                value = None
            if name and value is not None:
                rows.append((name, value))
        if rows:
            names = [r[0] for r in rows][:30]
            vals = [r[1] for r in rows][:30]
            fig = plt.figure()
            plt.bar(range(len(vals)), vals)
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.title("JSON -> Metrikler")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Sayısal metrik bulunamadı.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⬇️ Dışa Aktar")
        if active["json"]:
            def df_from_list_of_dicts(rows, columns):
                if not rows:
                    return None
                df = pd.DataFrame(rows)
                avail = [c for c in columns if c in df.columns]
                return df[avail] if avail else df

            def csv_bytes_from_df(df):
                return df.to_csv(index=False).encode("utf-8")

            sections = {
                "metrics": ["name", "value", "unit"],
                "anomalies": ["title", "where", "why"],
                "trends": ["signal", "metric", "confidence"],
                "quality_flags": ["issue", "severity"],
                "recommendations": ["title", "impact", "effort", "steps"],
            }
            for sec, cols in sections.items():
                rows_sec = active["json"].get(sec) or []
                df_sec = df_from_list_of_dicts(rows_sec, cols)
                if df_sec is not None:
                    st.download_button(
                        f"{sec}.csv",
                        data=csv_bytes_from_df(df_sec),
                        file_name=f"{sec}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            st.download_button(
                "Rapor (Markdown)",
                data=active["text"],
                file_name=f"{active['name']}_analysis.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("JSON verisi yok.")
        st.markdown("</div>", unsafe_allow_html=True)
