import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Insomnia Risk Predictor",
    layout="wide",
)

st.markdown("""
<style>
    .risk-box {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .risk-low  { background: #d4edda; color: #155724; }
    .risk-med  { background: #fff3cd; color: #856404; }
    .risk-high { background: #f8d7da; color: #721c24; }
    h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

FREQ_LABELS   = ['Never', 'Rarely (1-2x/week)', 'Sometimes (3-4x/week)', 'Often (5-6x/week)', 'Every day']
SLEEP_LABELS  = ['< 4 jam', '4–5 jam', '6–7 jam', '7–8 jam (optimal)', '> 8 jam']
STRESS_LABELS = ['No stress', 'Low stress', 'High stress', 'Extremely high stress']
SLEEP_MAP     = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0}  # hanya 7–8 jam = 1
FEATURE_COLS  = ['Sleep_Hours', 'Caffeine', 'Screen_Time', 'Exercise', 'Stress_Level']

@st.cache_resource(show_spinner="Memuat model…")
def load_artifacts():
    return joblib.load("Random_Forest_Model.pkl")

try:
    pipeline = load_artifacts()
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: `{e.filename}`\n\nPastikan `model.pkl` dan `scaler.pkl` ada di direktori yang sama dengan `streamlit_app.py`.")
    st.stop()

def predict_risk(profile: dict) -> float:
    df = pd.DataFrame([profile])[FEATURE_COLS]
    prob = pipeline.predict_proba(df)[0][1]
    return round(prob * 100, 1)

def risk_label(pct):
    if pct < 35:
        return "Risiko Rendah", "risk-low"
    elif pct < 65:
        return "Risiko Sedang", "risk-med"
    else:
        return "Risiko Tinggi", "risk-high"

st.sidebar.header("Profil Mahasiswa")

sleep_idx    = st.sidebar.select_slider("Durasi Tidur per Hari",      options=list(range(5)), format_func=lambda i: SLEEP_LABELS[i],  value=2)
caffeine_idx = st.sidebar.select_slider("Konsumsi Kafein",             options=list(range(5)), format_func=lambda i: FREQ_LABELS[i],   value=2)
screen_idx   = st.sidebar.select_slider("Screen Time Sebelum Tidur",  options=list(range(5)), format_func=lambda i: FREQ_LABELS[i],   value=3)
exercise_idx = st.sidebar.select_slider("Frekuensi Olahraga",         options=list(range(5)), format_func=lambda i: FREQ_LABELS[i],   value=1)
stress_idx   = st.sidebar.select_slider("Tingkat Stres Akademik",     options=list(range(4)), format_func=lambda i: STRESS_LABELS[i], value=2)

profile = {
    'Sleep_Hours':  SLEEP_MAP[sleep_idx],
    'Caffeine':     caffeine_idx,
    'Screen_Time':  screen_idx,
    'Exercise':     exercise_idx,
    'Stress_Level': stress_idx,
}

st.sidebar.markdown("---")
st.sidebar.caption("Model: Random Forest")
st.sidebar.caption("Loaded from pickle")

st.title("Insomnia Risk Predictor")
st.markdown("Demonstrasi model **Random Forest** untuk memprediksi risiko insomnia mahasiswa berdasarkan gaya hidup.")

tab1 = st.tabs(["Prediksi & Simulasi"])[0]

with tab1:
    base_risk = predict_risk(profile)
    label, css_class = risk_label(base_risk)

    col_risk, col_gauge = st.columns([1, 1])

    with col_risk:
        st.markdown(f"""
        <div class="risk-box {css_class}">
            <h2 style="margin:0">{label}</h2>
            <div style="font-size:3.5rem; font-weight:800; margin:8px 0">{base_risk}%</div>
            <div style="opacity:.75">Probabilitas Risiko Insomnia</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Profil saat ini:**")
        st.markdown(f"""
| Faktor | Nilai |
|---|---|
| Durasi Tidur | {SLEEP_LABELS[sleep_idx]} |
| Kafein | {FREQ_LABELS[caffeine_idx]} |
| Screen Time | {FREQ_LABELS[screen_idx]} |
| Olahraga | {FREQ_LABELS[exercise_idx]} |
| Stres | {STRESS_LABELS[stress_idx]} |
""")

    with col_gauge:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(0, 100, height=0.4, color='#e9ecef')
        pct_color = '#155724' if base_risk < 35 else '#856404' if base_risk < 65 else '#721c24'
        ax.barh(0, base_risk, height=0.4, color=pct_color)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        ax.text(min(base_risk + 1, 98), 0, f" {base_risk}%", va='center', fontweight='bold', fontsize=14, color=pct_color)
        ax.text(5,  -0.4, "Rendah", fontsize=8, color='#28a745')
        ax.text(40, -0.4, "Sedang", fontsize=8, color='#856404')
        ax.text(72, -0.4, "Tinggi", fontsize=8, color='#dc3545')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.subheader("What-If Analysis")
    st.caption("Simulasi dampak jika satu kebiasaan diubah menjadi lebih baik.")

    scenarios = {
        "Tambah Tidur":        {**profile, 'Sleep_Hours':  min(1, profile['Sleep_Hours'] + 1)},
        "Kurangi Kafein":       {**profile, 'Caffeine':     max(0, profile['Caffeine'] - 1)},
        "Kurangi Screen Time": {**profile, 'Screen_Time':  max(0, profile['Screen_Time'] - 1)},
        "Tambah Olahraga":     {**profile, 'Exercise':     min(4, profile['Exercise'] + 1)},
        "Kurangi Stres":       {**profile, 'Stress_Level': max(0, profile['Stress_Level'] - 1)},
    }

    results = sorted(
        [{'action': a, 'new_risk': predict_risk(p), 'improvement': round(base_risk - predict_risk(p), 1)}
         for a, p in scenarios.items()],
        key=lambda x: x['improvement'], reverse=True
    )
    best = results[0]

    cols = st.columns(len(results))
    for i, (col, r) in enumerate(zip(cols, results)):
        is_best   = (i == 0 and r['improvement'] > 0)
        border    = "2px solid #28a745" if is_best else "1px solid #dee2e6"
        bg        = "#f0fff4" if is_best else "#fafafa"
        imp_color = "#155724" if r['improvement'] > 0 else "#6c757d"
        badge     = " OK" if is_best else ""
        col.markdown(f"""
<div style="border:{border}; border-radius:10px; color: black; padding:14px; background:{bg}; text-align:center; height:130px">
    <div style="font-size:.82rem; font-weight:600; margin-bottom:6px">{r['action']}{badge}</div>
    <div style="font-size:1.6rem; font-weight:800">{r['new_risk']}%</div>
    <div style="font-size:.8rem; color:{imp_color}">{'−' if r['improvement']>0 else '±'}{abs(r['improvement'])}%</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    if best['improvement'] > 0:
        st.success(f"**Rekomendasi terbaik:** {best['action']} → risiko turun **{best['improvement']}%** menjadi **{best['new_risk']}%**")
    else:
        st.info("Tidak ada intervensi tunggal yang signifikan. Coba kombinasikan beberapa perubahan sekaligus.")
