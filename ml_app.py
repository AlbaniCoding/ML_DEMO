import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Buyer's Remorse Predictor", layout="centered")

# ── Session state init ────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

# ── Constants ─────────────────────────────────────────────────────────────────
options_map = {
    -2: "Strongly Disagree",
    -1: "Disagree",
     0: "Neither Agree nor Disagree",
     1: "Agree",
     2: "Strongly Agree",
}

PAGE_ORDER  = ["home", "demographics", "cd", "sat", "pr", "result"]
PAGE_LABELS = {
    "home":         "Home",
    "demographics": "Profil",
    "cd":           "Cognitive Dissonance",
    "sat":          "Satisfaction",
    "pr":           "Post-purchase Regret",
    "result":       "Result",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def go_to(page: str):
    st.session_state.page = page

def render_progress():
    current_idx = PAGE_ORDER.index(st.session_state.page)
    # Skip "home" from the progress bar
    visible_pages = PAGE_ORDER[1:]
    visible_labels = {k: v for k, v in PAGE_LABELS.items() if k != "home"}
    cols = st.columns(len(visible_pages))
    for i, (page, label) in enumerate(visible_labels.items()):
        idx_in_order = PAGE_ORDER.index(page)
        with cols[i]:
            if idx_in_order < current_idx:
                st.markdown(
                    f"<div style='text-align:center;color:#4CAF50;font-size:11px'>✓<br>{label}</div>",
                    unsafe_allow_html=True,
                )
            elif idx_in_order == current_idx:
                st.markdown(
                    f"<div style='text-align:center;color:#1E90FF;font-weight:bold;font-size:11px'>●<br>{label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center;color:#aaa;font-size:11px'>○<br>{label}</div>",
                    unsafe_allow_html=True,
                )
    st.divider()

def render_slider_section(questions: dict, parameter: dict):
    """Render sliders for a set of questions and store text label in parameter."""
    for key, label in questions.items():
        val = st.slider(label, min_value=-2, max_value=2, value=0, key=f"slider_{key}")
        parameter[key] = options_map[val]
        st.caption(f"↳ {parameter[key]}")
        st.write("")

# ── Metadata hardcoded (mirrors training) — no encoder objects needed ─────────
GENDER_CLASSES  = ["Female", "Male"]
AGE_ORDER       = ["22-30", "31 - 40", "41 - 50", "Over 50"]
EDU_CLASSES     = ["Master student", "Other", "Ph.D student", "postgraduate"]
INCOME_ORDER    = ["2,000 to 3,000", "3,000 to 4,000", "5,000 to 7,000", "7,000 or greater"]
MARITAL_CLASSES = ["Marrried", "Single"]

LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree":          2,
    "Neither Agree nor Disagree": 3,
    "Agree":             4,
    "Strongly Agree":    5,
}

@st.cache_resource
def load_model():
    """Load pkl — handles both bare DecisionTreeClassifier and bundle dict."""
    with open("dt_model.pkl", "rb") as f:
        obj = pickle.load(f)
    # If it's a raw model (old format), wrap it
    if not isinstance(obj, dict):
        return {"model": obj}
    return obj

def run_prediction(bundle: dict) -> tuple[str, float, dict]:
    """
    Build feature vector from session state and run the model.
    Returns (predicted_label, confidence, scores_dict).
    """
    model = bundle["model"]

    # ── Compute Likert composite scores ──────────────────────────────────────
    def mean_score(keys):
        return np.mean([LIKERT_MAP[st.session_state[k]] for k in keys])

    cd_score  = mean_score(["CD1","CD2","CD3","CD4","CD5","CD6","CD7","CD8"])
    sat_score = mean_score(["SAT1","SAT2","SAT3"])
    pr_score  = mean_score(["PR1","PR2","PR3","PR4"])

    # ── Encode demographics with index-based ordinal encoding ─────────────────
    gender_enc  = GENDER_CLASSES.index(st.session_state["Gender"])
    age_enc     = AGE_ORDER.index(st.session_state["Age"])
    edu_enc     = EDU_CLASSES.index(st.session_state["Education"])
    income_enc  = INCOME_ORDER.index(st.session_state["Monthly Income"])
    marital_enc = MARITAL_CLASSES.index(st.session_state["Marital Status"])

    X = np.array([[gender_enc, age_enc, edu_enc, income_enc, marital_enc,
                   cd_score, sat_score, pr_score]])

    pred       = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    confidence = proba[pred]

    label = "High Switching" if pred == 1 else "Low Switching"
    scores = {
        "CD Score":  round(cd_score, 2),
        "SAT Score": round(sat_score, 2),
        "PR Score":  round(pr_score, 2),
    }
    return label, confidence, scores


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

# ── HOME ──────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.title("Buyer's Remorse Predictor")
    st.subheader("Apakah Anda akan menyesal setelah membeli?")
    st.write(
        "Aplikasi ini membantu memprediksi apakah Anda akan merasa puas atau menyesal "
        "dengan keputusan pembelian Anda, berdasarkan tiga dimensi psikologis:"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Cognitive Dissonance**\n\nKetidaksesuaian antara harapan dan kenyataan")
    with col2:
        st.info("**Satisfaction**\n\nTingkat kepuasan terhadap pembelian")
    with col3:
        st.info("**Post-purchase Regret**\n\nRasa penyesalan setelah membeli")

    st.write("")
    st.write("Kuesioner ini terdiri dari **18 pertanyaan** dan memakan waktu sekitar **3–4 menit**.")

    if st.button("Mulai Kuesioner →", type="primary", use_container_width=True):
        go_to("demographics")
        st.rerun()

# ── DEMOGRAPHICS ──────────────────────────────────────────────────────────────
elif st.session_state.page == "demographics":
    render_progress()
    st.header("Profil Responden")
    st.write("Lengkapi data diri Anda sebelum mengisi kuesioner.")
    st.write("")

    gender = st.selectbox(
        "Jenis Kelamin",
        options=GENDER_CLASSES,
        index=0,
        key="sel_gender",
    )
    age = st.selectbox(
        "Usia",
        options=AGE_ORDER,
        index=0,
        key="sel_age",
    )
    education = st.selectbox(
        "Pendidikan Terakhir",
        options=EDU_CLASSES,
        index=0,
        key="sel_edu",
    )
    income = st.selectbox(
        "Pendapatan Bulanan (USD)",
        options=INCOME_ORDER,
        index=0,
        key="sel_income",
    )
    marital = st.selectbox(
        "Status Pernikahan",
        options=MARITAL_CLASSES,
        index=0,
        key="sel_marital",
    )

    st.write("")
    if st.button("Lanjut ke Cognitive Dissonance →", type="primary", use_container_width=True):
        st.session_state["Gender"]         = gender
        st.session_state["Age"]            = age
        st.session_state["Education"]      = education
        st.session_state["Monthly Income"] = income
        st.session_state["Marital Status"] = marital
        go_to("cd")
        st.rerun()

# ── COGNITIVE DISSONANCE ──────────────────────────────────────────────────────
elif st.session_state.page == "cd":
    render_progress()
    st.header("Cognitive Dissonance")
    st.write("Jawab setiap pernyataan berikut sesuai dengan perasaan Anda terkait pembelian terakhir Anda.")
    st.write("")

    cd_questions = {
        "CD1": "Saya merasa produk yang saya beli sesuai dengan yang saya harapkan.",
        "CD2": "Saya merasa nyaman dengan keputusan pembelian saya.",
        "CD3": "Saya tidak merasa ragu setelah melakukan pembelian ini.",
        "CD4": "Saya merasa informasi yang saya miliki sebelum membeli sudah cukup.",
        "CD5": "Saya tidak merasa tertekan untuk membeli produk ini.",
        "CD6": "Saya merasa produk ini memenuhi kebutuhan saya.",
        "CD7": "Saya tidak merasa menyesal memilih produk ini dibanding alternatif lain.",
        "CD8": "Saya merasa keputusan pembelian ini rasional dan tidak impulsif.",
    }

    parameter = {}
    render_slider_section(cd_questions, parameter)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Kembali", use_container_width=True):
            go_to("demographics")
            st.rerun()
    with col2:
        if st.button("Lanjut ke Satisfaction →", type="primary", use_container_width=True):
            for k, v in parameter.items():
                st.session_state[k] = v
            go_to("sat")
            st.rerun()

# ── SATISFACTION ──────────────────────────────────────────────────────────────
elif st.session_state.page == "sat":
    render_progress()
    st.header("Satisfaction")
    st.write("Seberapa puas Anda dengan pembelian yang telah dilakukan?")
    st.write("")

    sat_questions = {
        "SAT1": "Secara keseluruhan, saya puas dengan produk yang saya beli.",
        "SAT2": "Produk ini memberikan nilai yang sebanding dengan harganya.",
        "SAT3": "Saya akan merekomendasikan produk ini kepada orang lain.",
    }

    parameter = {}
    render_slider_section(sat_questions, parameter)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Kembali", use_container_width=True):
            go_to("cd")
            st.rerun()
    with col2:
        if st.button("Lanjut ke Post-purchase Regret →", type="primary", use_container_width=True):
            for k, v in parameter.items():
                st.session_state[k] = v
            go_to("pr")
            st.rerun()

# ── POST-PURCHASE REGRET ──────────────────────────────────────────────────────
elif st.session_state.page == "pr":
    render_progress()
    st.header("Post-purchase Regret")
    st.write("Bagaimana perasaan Anda setelah beberapa waktu berlalu sejak pembelian?")
    st.write("")

    pr_questions = {
        "PR1": "Saya merasa menyesal telah membeli produk ini.",
        "PR2": "Saya berharap saya tidak jadi membeli produk ini.",
        "PR3": "Saya merasa uang yang saya keluarkan tidak sepadan.",
        "PR4": "Jika bisa mengulang, saya tidak akan membeli produk ini lagi.",
    }

    parameter = {}
    render_slider_section(pr_questions, parameter)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Kembali", use_container_width=True):
            go_to("sat")
            st.rerun()
    with col2:
        if st.button("Lihat Hasil →", type="primary", use_container_width=True):
            for k, v in parameter.items():
                st.session_state[k] = v
            go_to("result")
            st.rerun()

# ── RESULT ────────────────────────────────────────────────────────────────────
elif st.session_state.page == "result":
    render_progress()
    st.header("Hasil Analisis")

    # Verify all required keys are present
    required = ["Gender","Age","Education","Monthly Income","Marital Status",
                "CD1","CD2","CD3","CD4","CD5","CD6","CD7","CD8",
                "SAT1","SAT2","SAT3","PR1","PR2","PR3","PR4"]
    missing = [k for k in required if k not in st.session_state]

    if missing:
        st.error(f"Data tidak lengkap: {missing}. Silakan ulangi kuesioner.")
        if st.button("Ulangi dari Awal", use_container_width=True):
            go_to("home")
            st.rerun()
    else:
        bundle = load_model()
        label, confidence, scores = run_prediction(bundle)

        # ── Result card ───────────────────────────────────────────────────────
        is_high = label == "High Switching"
        color   = "#FF4B4B" if is_high else "#4CAF50"
        emoji   = "⚠️" if is_high else "✅"
        title   = "Kecenderungan Beralih TINGGI" if is_high else "Kecenderungan Beralih RENDAH"
        desc    = (
            "Berdasarkan jawaban Anda, Anda menunjukkan kecenderungan **tinggi** untuk "
            "berpindah ke produk/merek lain. Tingkat kepuasan atau keselarasan kognitif "
            "Anda terhadap pembelian ini relatif rendah."
            if is_high else
            "Berdasarkan jawaban Anda, Anda menunjukkan kecenderungan **rendah** untuk "
            "berpindah ke produk/merek lain. Anda tampak cukup puas dan nyaman dengan "
            "keputusan pembelian ini."
        )

        st.markdown(
            f"""
            <div style="border:2px solid {color};border-radius:12px;padding:20px 24px;
                        background-color:{color}18;margin-bottom:16px;">
                <h2 style="color:{color};margin:0">{emoji} {title}</h2>
                <p style="color:#ccc;margin-top:8px;margin-bottom:0">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Confidence bar ────────────────────────────────────────────────────
        st.write("")
        st.markdown(f"**Confidence Model:** `{confidence:.1%}`")
        st.progress(float(confidence))

        # ── Score breakdown ───────────────────────────────────────────────────
        st.write("")
        st.subheader("Breakdown Skor")
        st.caption("Skor berkisar dari 1 (sangat negatif) hingga 5 (sangat positif)")

        col1, col2, col3 = st.columns(3)
        score_meta = {
            "CD Score":  ("Cognitive Dissonance", "Skor tinggi = sedikit disonansi"),
            "SAT Score": ("Satisfaction",          "Skor tinggi = sangat puas"),
            "PR Score":  ("Post-purchase Regret",  "Skor tinggi = banyak penyesalan"),
        }
        for col, (key, (name, hint)) in zip([col1, col2, col3], score_meta.items()):
            val = scores[key]
            # Normalize 1–5 → 0–1 for progress bar
            with col:
                st.metric(label=name, value=f"{val:.2f} / 5")
                st.progress((val - 1) / 4)
                st.caption(hint)

        # ── Demographics summary ──────────────────────────────────────────────
        st.write("")
        with st.expander("📋 Data Profil yang Digunakan"):
            demo_cols = ["Gender","Age","Education","Monthly Income","Marital Status"]
            for k in demo_cols:
                st.write(f"**{k}:** {st.session_state.get(k, '-')}")

        st.divider()
        if st.button("🔄 Ulangi Kuesioner", use_container_width=True):
            all_keys = required + ["Gender","Age","Education","Monthly Income","Marital Status"]
            for k in all_keys:
                if k in st.session_state:
                    del st.session_state[k]
            go_to("home")
            st.rerun()
