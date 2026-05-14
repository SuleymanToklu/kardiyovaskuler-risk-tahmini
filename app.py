"""
app.py
------
Makale reprodüksiyonu projesi — Adım 6: Streamlit Arayüzü (Dashboard).

Sekmeler:
  🔍 Risk Tahmini   — Bulanık mantık kural motoru
  📊 EDA            — Keşifsel veri analizi grafikleri
  🤖 Model Sonuçları — Karşılaştırma tablosu + confusion matrix + ROC eğrisi

Kullanım:
  streamlit run app.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Sayfa yapılandırması
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Kardiyovasküler Risk Tahmini",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Özel CSS — Premium görünüm
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Arka plan ── */
    .stApp {
        background: radial-gradient(ellipse at 15% 60%, rgba(0,212,184,0.11) 0%, transparent 55%),
                    radial-gradient(ellipse at 85% 15%, rgba(249,115,22,0.10) 0%, transparent 55%),
                    linear-gradient(160deg, #060d0b 0%, #0b1714 40%, #0f1e1a 100%);
        min-height: 100vh;
    }

    /* ── Hero ── */
    .hero-wrap { text-align: center; padding: 1.8rem 0 1rem 0; }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #ffffff 0%, #5eead4 45%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleGlow 4s ease-in-out infinite alternate;
        margin-bottom: 0.3rem;
    }
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 18px rgba(0,212,184,0.30)); }
        to   { filter: drop-shadow(0 0 36px rgba(249,115,22,0.40)); }
    }
    .heart-pulse {
        display: inline-block;
        animation: heartbeat 2s ease-in-out infinite;
        -webkit-text-fill-color: initial;
        background: none;
        filter: none;
    }
    @keyframes heartbeat {
        0%,100% { transform: scale(1);    }
        14%     { transform: scale(1.18); }
        28%     { transform: scale(1);    }
        42%     { transform: scale(1.10); }
        56%     { transform: scale(1);    }
    }
    .hero-sub {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.45);
        margin-bottom: 0.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(0,188,164,0.85), rgba(249,115,22,0.85));
        color: white;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1.4rem;
        box-shadow: 0 2px 14px rgba(0,212,184,0.35);
    }

    /* ── Sekmeler ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 14px !important;
        padding: 5px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: rgba(255,255,255,0.45) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 22px !important;
        transition: all 0.2s ease !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: rgba(255,255,255,0.75) !important;
        background: rgba(255,255,255,0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,188,164,0.45), rgba(249,115,22,0.35)) !important;
        color: #ffffff !important;
        box-shadow: 0 0 18px rgba(0,212,184,0.30), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* ── Bölüm etiketi ── */
    .section-label {
        font-size: 0.73rem;
        font-weight: 700;
        color: #2dd4bf;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        margin-bottom: 0.9rem;
        margin-top: 0.4rem;
    }
    .section-label::after {
        content: '';
        display: block;
        height: 2px;
        width: 28px;
        margin-top: 5px;
        border-radius: 2px;
        background: linear-gradient(90deg, #00d4b8, #f97316);
        box-shadow: 0 0 6px rgba(0,212,184,0.55);
    }

    /* ── Çizgi ── */
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin: 1.4rem 0;
    }

    /* ── Form konteyneri ── */
    [data-testid="stForm"] {
        background: rgba(255,255,255,0.035) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    }

    /* ── Input alanları ── */
    .stSelectbox label, .stNumberInput label, .stRadio label {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: white !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stSelectbox > div > div:hover {
        border-color: rgba(0,212,184,0.55) !important;
        box-shadow: 0 0 0 3px rgba(0,212,184,0.12) !important;
    }
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: rgba(0,212,184,0.55) !important;
        box-shadow: 0 0 0 3px rgba(0,212,184,0.14) !important;
    }
    div[data-testid="stRadio"] > div {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 10px !important;
        padding: 0.4rem 0.6rem !important;
        gap: 6px !important;
    }

    /* ── Sonuç kutuları ── */
    .result-box {
        border-radius: 16px;
        padding: 1.7rem 2rem;
        margin-top: 1.2rem;
        animation: fadeSlide 0.45s cubic-bezier(.16,1,.3,1) both;
        border: 1px solid;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    .result-green {
        background: rgba(0,212,184,0.09);
        border-color: rgba(0,212,184,0.45);
        box-shadow: 0 0 30px rgba(0,212,184,0.20), inset 0 1px 0 rgba(0,212,184,0.1);
    }
    .result-yellow {
        background: rgba(249,115,22,0.09);
        border-color: rgba(249,115,22,0.45);
        box-shadow: 0 0 30px rgba(249,115,22,0.20), inset 0 1px 0 rgba(249,115,22,0.1);
    }
    .result-red {
        background: rgba(239,68,68,0.09);
        border-color: rgba(239,68,68,0.45);
        box-shadow: 0 0 30px rgba(239,68,68,0.20), inset 0 1px 0 rgba(239,68,68,0.1);
    }
    .result-title { font-size: 1.35rem; font-weight: 800; margin-bottom: 0.35rem; }
    .result-desc  { font-size: 0.88rem; opacity: 0.84; line-height: 1.6; }
    .result-meta  { font-size: 0.76rem; opacity: 0.55; margin-top: 0.8rem; font-family: monospace; }

    @keyframes fadeSlide {
        from { opacity: 0; transform: translateY(12px) scale(0.98); }
        to   { opacity: 1; transform: translateY(0)  scale(1); }
    }

    /* ── Metric kartları ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.4rem !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 32px rgba(0,212,184,0.22) !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: rgba(255,255,255,0.5) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #2dd4bf !important;
        font-weight: 800 !important;
        font-size: 1.75rem !important;
    }

    /* ── Buton ── */
    .stButton > button {
        background: linear-gradient(135deg, #00bfa8, #f97316) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        width: 100% !important;
        letter-spacing: 0.4px !important;
        transition: transform 0.18s, box-shadow 0.18s !important;
        box-shadow: 0 4px 20px rgba(0,191,168,0.45), 0 0 40px rgba(249,115,22,0.15) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,191,168,0.60), 0 0 60px rgba(249,115,22,0.25) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* ── Kaydırma çubuğu ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,212,184,0.4); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(249,115,22,0.55); }

    /* ── Genel metin ── */
    .stMarkdown p, .stMarkdown strong { color: rgba(255,255,255,0.85); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Veritabanı yardımcıları
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent / "heart_disease.db"

_PLOTLY_THEME = "plotly_dark"


def _read_table(table: str) -> pd.DataFrame | None:
    if not DB_PATH.is_file():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_model_results() -> pd.DataFrame | None:
    return _read_table("model_results")


@st.cache_data(show_spinner=False)
def load_confusion_matrices() -> pd.DataFrame | None:
    return _read_table("confusion_matrices")


@st.cache_data(show_spinner=False)
def load_roc_curves() -> pd.DataFrame | None:
    return _read_table("roc_curves")


@st.cache_data(show_spinner=False)
def load_cleaned_data() -> pd.DataFrame | None:
    if not DB_PATH.is_file():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query("SELECT * FROM cleaned_data", conn, index_col="id")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_eda_correlation() -> pd.DataFrame | None:
    return _read_table("eda_correlation")


# ---------------------------------------------------------------------------
# Karar Motoru (Tablo 13 Kural Seti)
# ---------------------------------------------------------------------------

AGE_THRESHOLD = {1: 55, 2: 45}


def compute_other_factors(smoke: int, alco: int, active: int) -> float:
    if smoke == 0 and alco == 0 and active == 1:
        return 0.0
    if smoke == 1 and alco == 1 and active == 0:
        return 1.0
    return 0.5


def evaluate_risk(age: int, gender: int, other_factors: float):
    threshold = AGE_THRESHOLD[gender]
    if age < threshold and other_factors == 0.0:
        return (
            0.0, "✅  Risk Yok", "result-green",
            f"Yaşınız ({age}) yaş eşiğinin ({threshold}) altında ve yaşam "
            f"alışkanlıklarınız tamamen sağlıklı. Kardiyovasküler hastalık riski düşük görünüyor.",
        )
    if age >= threshold and other_factors == 1.0:
        return (
            1.0, "🔴  Risk Var", "result-red",
            f"Yaşınız ({age}) yaş eşiğini ({threshold}) aşmış ve yaşam alışkanlıklarınız "
            f"tamamen sağlıksız. Lütfen bir kardiyolog ile görüşün.",
        )
    return (
        0.5, "⚠️  Risk Olabilir", "result-yellow",
        f"Yaşınız ({age}) ve/veya alışkanlık profiliniz belirsizlik içeriyor "
        f"(yaş eşiği: {threshold}). Düzenli sağlık kontrolü önerilir.",
    )


# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">
            <span class="heart-pulse">🫀</span> Kardiyovasküler Risk Tahmini
        </div>
        <div class="hero-sub">Bulanık Mantık Tabanlı Kural Motoru &nbsp;·&nbsp; 68 601 Hasta · 7 Model</div>
        <div style="margin-top:0.7rem;">
            <span class="hero-badge">Makale Reprodüksiyonu &nbsp;·&nbsp; Tablo 13</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sekmeler
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["🔍  Risk Tahmini", "📊  EDA", "🤖  Model Sonuçları"])

# ═══════════════════════════════════════════════════════════════════════════
# SEKME 1 — Risk Tahmini
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form(key="risk_form"):
        st.markdown('<div class="section-label">👤 Kişisel Bilgiler</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Yaş", min_value=18, max_value=100, value=50, step=1,
                                  help="Yıl cinsinden yaşınızı girin (18–100)")
        with col2:
            gender_label = st.selectbox("Cinsiyet", options=["Kadın", "Erkek"], index=0)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🚬 Yaşam Alışkanlıkları</div>', unsafe_allow_html=True)

        col3, col4, col5 = st.columns(3)
        with col3:
            smoke_label  = st.radio("Sigara içiyor mu?",        options=["Hayır", "Evet"], index=0)
        with col4:
            alco_label   = st.radio("Alkol kullanıyor mu?",     options=["Hayır", "Evet"], index=0)
        with col5:
            active_label = st.radio("Fiziksel olarak aktif mi?", options=["Evet", "Hayır"], index=0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Riski Hesapla")

    if submitted:
        gender_code = 1 if gender_label == "Kadın" else 2
        smoke_code  = 1 if smoke_label  == "Evet"  else 0
        alco_code   = 1 if alco_label   == "Evet"  else 0
        active_code = 1 if active_label == "Evet"  else 0

        of_val = compute_other_factors(smoke_code, alco_code, active_code)
        risk_score, label, css_cls, explanation = evaluate_risk(age, gender_code, of_val)

        of_label_map   = {0.0: "0 — Tümü Sağlıklı", 0.5: "0.5 — Belirsiz/Karışık", 1.0: "1 — Tümü Sağlıksız"}
        risk_label_map = {0.0: "0 — Risk Yok", 0.5: "0.5 — Risk Olabilir", 1.0: "1 — Risk Var"}

        st.markdown(
            f"""
            <div class="result-box {css_cls}">
                <div class="result-title">{label}</div>
                <div class="result-desc">{explanation}</div>
                <div class="result-meta">
                    other_factors = {of_label_map[of_val]} &nbsp;|&nbsp;
                    risk_score = {risk_label_map[risk_score]} &nbsp;|&nbsp;
                    yaş_eşiği = {AGE_THRESHOLD[gender_code]} yıl
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📋 Giriş Detayları"):
            detail_df = pd.DataFrame({
                "Parametre": ["Yaş", "Cinsiyet", "Sigara", "Alkol", "Aktif", "other_factors", "Risk Skoru"],
                "Ham Değer": [age, gender_label, smoke_label, alco_label, active_label, "—", "—"],
                "Kodlanmış": [age, gender_code, smoke_code, alco_code, active_code, of_val, risk_score],
            })
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# SEKME 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Keşifsel Veri Analizi</div>', unsafe_allow_html=True)

    df_clean = load_cleaned_data()

    if df_clean is None:
        st.info("Temizlenmiş veri bulunamadı. Önce `preprocessing.py` scriptini çalıştırın.")
    else:
        # ── Özet metrikler ──────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Toplam Gözlem", f"{len(df_clean):,}")
        m2.metric("Özellik Sayısı", len(df_clean.columns))
        m3.metric("Eksik Değer", int(df_clean.isnull().sum().sum()))
        cardio_pct = (df_clean["cardio"].sum() / len(df_clean) * 100) if "cardio" in df_clean.columns else 0
        m4.metric("Kardiyovasküler (+)", f"{cardio_pct:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Cardio dağılımı + Özellik histogram ─────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Hedef Değişken Dağılımı (cardio)**")
            if "cardio" in df_clean.columns:
                dist = df_clean["cardio"].value_counts().reset_index()
                dist.columns = ["cardio", "count"]
                dist["Etiket"] = dist["cardio"].map({0: "Sağlıklı (0)", 1: "Kardiyovasküler (1)"})
                fig_pie = px.pie(
                    dist, names="Etiket", values="count",
                    color_discrete_sequence=["#48cae4", "#ef4444"],
                    template=_PLOTLY_THEME,
                    hole=0.4,
                )
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.markdown("**Özellik Dağılımı**")
            num_cols = df_clean.select_dtypes(include="number").columns.tolist()
            sel_col  = st.selectbox("Sütun seç:", num_cols, key="eda_col_select")

            n_unique = df_clean[sel_col].nunique()
            sample   = df_clean[sel_col].sample(min(5000, len(df_clean)), random_state=42)

            if n_unique <= 6:
                fig_dist = px.histogram(
                    sample, x=sel_col, text_auto=True,
                    color_discrete_sequence=["#6c63ff"],
                    template=_PLOTLY_THEME,
                    category_orders={sel_col: sorted(sample.unique().tolist())},
                )
            else:
                fig_dist = px.histogram(
                    sample, x=sel_col, nbins=40,
                    color_discrete_sequence=["#6c63ff"],
                    template=_PLOTLY_THEME,
                )
            fig_dist.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300,
                                   bargap=0.05)
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Korelasyon ısı haritası ──────────────────────────────────────
        st.markdown("**Pearson Korelasyon Matrisi**")
        corr_df = load_eda_correlation()

        if corr_df is not None:
            corr_matrix = corr_df.set_index("column")
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=".2f",
                template=_PLOTLY_THEME,
                aspect="auto",
            )
            fig_corr.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=420,
                coloraxis_colorbar=dict(title="r"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Korelasyon verisi bulunamadı. Önce `eda.py` scriptini çalıştırın.")

# ═══════════════════════════════════════════════════════════════════════════
# SEKME 3 — Model Sonuçları
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    results_df = load_model_results()

    if results_df is None or results_df.empty:
        st.info("Henüz model sonucu bulunamadı. Önce `model_training.py` scriptini çalıştırın.")
    else:
        # ── Karşılaştırma tablosu ────────────────────────────────────────
        st.markdown('<div class="section-label">📋 Karşılaştırma Tablosu</div>', unsafe_allow_html=True)

        # Sütun adlarını kullanıcı dostu hale getir
        col_rename = {
            "Classifier"        : "Model",
            "Accuracy"          : "Accuracy",
            "Macro_Precision"   : "Macro-Prec",
            "Macro_F1"          : "Macro-F1",
            "ROC_AUC"           : "ROC-AUC",
            "CV_Acc_Mean"       : "CV-Acc (Ort)",
            "CV_Acc_Std"        : "CV-Acc (Std)",
            "Computation_Time_s": "Süre (s)",
        }

        for scenario in ["Normal", "Fuzzy"]:
            grp = results_df[results_df["Scenario"] == scenario].copy()
            if grp.empty:
                continue

            grp = grp.drop(columns="Scenario").reset_index(drop=True)
            grp = grp.rename(columns=col_rename)
            grp = grp.sort_values("Accuracy", ascending=False)

            label = (
                "🔵 Normal Senaryo (Binary — smoke/alco/active dahil)"
                if scenario == "Normal"
                else "🟣 Fuzzy Senaryo (Multiclass 0/1/2 — gender+age+other_factors)"
            )
            st.markdown(f"**{label}**")

            fmt_cols = {c: "{:.4f}" for c in ["Accuracy", "Macro-Prec", "Macro-F1", "Süre (s)"]}
            if "CV-Acc (Ort)" in grp.columns:
                fmt_cols["CV-Acc (Ort)"] = "{:.4f}"
                fmt_cols["CV-Acc (Std)"] = "{:.4f}"
            if "ROC-AUC" in grp.columns:
                fmt_cols["ROC-AUC"] = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"

            grad_cols = [c for c in ["Accuracy", "Macro-F1"] if c in grp.columns]

            styler = grp.style.format(fmt_cols)
            for c in grad_cols:
                styler = styler.background_gradient(subset=[c], cmap="RdYlGn")

            st.dataframe(styler, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # ── Karmaşıklık Matrisi ──────────────────────────────────────────
        st.markdown('<div class="section-label">🔲 Karmaşıklık Matrisi</div>', unsafe_allow_html=True)

        cm_df = load_confusion_matrices()

        if cm_df is not None and not cm_df.empty:
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                cm_scenario = st.selectbox(
                    "Senaryo:", results_df["Scenario"].unique().tolist(), key="cm_scenario"
                )
            with cm_col2:
                cm_clfs = results_df[results_df["Scenario"] == cm_scenario]["Classifier"].tolist()
                cm_clf  = st.selectbox("Sınıflandırıcı:", cm_clfs, key="cm_clf")

            row = cm_df[(cm_df["Scenario"] == cm_scenario) & (cm_df["Classifier"] == cm_clf)]
            if not row.empty:
                matrix = json.loads(row.iloc[0]["Matrix"])
                labels = json.loads(row.iloc[0]["Labels"])
                str_labels = [str(l) for l in labels]

                fig_cm = px.imshow(
                    matrix,
                    x=str_labels,
                    y=str_labels,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    template=_PLOTLY_THEME,
                    labels={"x": "Tahmin", "y": "Gerçek", "color": "Sayı"},
                )
                fig_cm.update_layout(
                    title=f"{cm_scenario} / {cm_clf}",
                    margin=dict(t=40, b=20, l=20, r=20),
                    height=350,
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info(
                "Karmaşıklık matrisi verisi bulunamadı. "
                "Güncellenmiş `model_training.py` scriptini çalıştırın."
            )

        # ── ROC Eğrisi (Normal / Binary senaryo) ────────────────────────
        st.markdown('<div class="section-label">📈 ROC Eğrisi (Normal Senaryo)</div>',
                    unsafe_allow_html=True)

        roc_df = load_roc_curves()

        if roc_df is not None and not roc_df.empty:
            roc_normal = roc_df[roc_df["Scenario"] == "Normal"]
            if not roc_normal.empty:
                roc_clfs = roc_normal["Classifier"].tolist()
                sel_clfs = st.multiselect(
                    "Sınıflandırıcı(lar):", roc_clfs, default=roc_clfs[:3], key="roc_clf"
                )

                fig_roc = go.Figure()
                fig_roc.add_shape(
                    type="line", x0=0, y0=0, x1=1, y1=1,
                    line=dict(color="gray", dash="dash", width=1),
                )

                colors = px.colors.qualitative.Plotly
                for i, clf_name in enumerate(sel_clfs):
                    r = roc_normal[roc_normal["Classifier"] == clf_name]
                    if r.empty:
                        continue
                    fpr  = json.loads(r.iloc[0]["FPR"])
                    tpr  = json.loads(r.iloc[0]["TPR"])
                    auc  = r.iloc[0]["AUC"]
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode="lines",
                        name=f"{clf_name} (AUC={auc:.4f})",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))

                fig_roc.update_layout(
                    template=_PLOTLY_THEME,
                    xaxis_title="Yanlış Pozitif Oranı (FPR)",
                    yaxis_title="Doğru Pozitif Oranı (TPR)",
                    legend=dict(font=dict(size=11)),
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=420,
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Normal senaryo ROC verisi bulunamadı.")
        else:
            st.info(
                "ROC eğrisi verisi bulunamadı. "
                "Güncellenmiş `model_training.py` scriptini çalıştırın."
            )

# ---------------------------------------------------------------------------
# Alt bilgi
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="text-align:center; margin-top:2rem; color:rgba(255,255,255,0.3);
                font-size:0.78rem; letter-spacing:0.3px;">
        Bulanık Mantık Tabanlı Kardiyovasküler Hastalık Riski Sınıflandırması &nbsp;·&nbsp;
        Makale Reprodüksiyonu
    </div>
    """,
    unsafe_allow_html=True,
)
