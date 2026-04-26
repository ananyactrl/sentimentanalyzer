import streamlit as st
import numpy as np
import re
import pickle
from sklearn.metrics import accuracy_score, classification_report

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyser | Ananya Singh",
    page_icon="🎬",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0c10;
    }
    .header-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .header-sub {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }
    .tag {
        display: inline-block;
        background: #1c2333;
        color: #60a5fa;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'Space Mono', monospace;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #1e3a5f;
    }
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1c2333 100%);
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Space Mono', monospace;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f9fafb;
        margin-top: 4px;
    }
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #e5e7eb !important;
        border: 1px solid #1f2937 !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
    }
    .result-positive {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border: 1px solid #16a34a;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #1c0505 0%, #450a0a 100%);
        border: 1px solid #dc2626;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .result-conf {
        font-size: 0.85rem;
        margin-top: 6px;
        opacity: 0.8;
    }
    .word-chip-pos {
        display: inline-block;
        background: #052e16;
        border: 1px solid #16a34a;
        color: #4ade80;
        font-size: 0.78rem;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-family: 'Space Mono', monospace;
    }
    .word-chip-neg {
        display: inline-block;
        background: #1c0505;
        border: 1px solid #dc2626;
        color: #f87171;
        font-size: 0.78rem;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-family: 'Space Mono', monospace;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.3px !important;
        padding: 10px 20px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
        transform: translateY(-1px) !important;
    }
    .example-btn > button {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        color: #9ca3af !important;
        font-size: 0.78rem !important;
        padding: 6px 12px !important;
    }
    .divider {
        border: none;
        border-top: 1px solid #1f2937;
        margin: 20px 0;
    }
    .footer {
        text-align: center;
        font-size: 0.75rem;
        color: #374151;
        padding: 20px 0 10px;
        font-family: 'Space Mono', monospace;
    }
    .pipeline-step {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: #9ca3af;
        text-align: center;
    }
    .arrow { color: #374151; font-size: 1.2rem; text-align: center; padding-top: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Load trained model ────────────────────────────────────
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ── Preprocess ────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="header-title">🎬 Sentiment<br>Analyser</div>

<br>
<span class="tag">Ananya Singh</span>
<span class="tag">PRN: 202301100050</span>
<span class="tag">TF-IDF + Logistic Regression</span>
<span class="tag">IMDB Dataset</span>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────
try:
    model, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("⚠️ Model files not found. Place `sentiment_model.pkl` and `tfidf_vectorizer.pkl` in the same folder as `app.py`.")

if model_loaded:

    # ── Metric cards ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value" style="font-size:1rem">Log. Reg.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-label">Features</div><div class="metric-value" style="font-size:1rem">TF-IDF</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">86.7%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-label">Dataset</div><div class="metric-value" style="font-size:1rem">IMDB</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline visual ───────────────────────────────────
    with st.expander("📐 Model Pipeline", expanded=False):
        p1, pa, p2, pb, p3, pc, p4 = st.columns([3,1,3,1,3,1,3])
        with p1: st.markdown('<div class="pipeline-step">📝 Raw Text Input</div>', unsafe_allow_html=True)
        with pa: st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
        with p2: st.markdown('<div class="pipeline-step">🧹 Preprocessing<br><small>lower, punct removal</small></div>', unsafe_allow_html=True)
        with pb: st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
        with p3: st.markdown('<div class="pipeline-step">📊 TF-IDF<br><small>15k features, bigrams</small></div>', unsafe_allow_html=True)
        with pc: st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
        with p4: st.markdown('<div class="pipeline-step">🤖 Logistic Regression<br><small>C=1.0, IMDB trained</small></div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────
    st.markdown("#### Enter a Movie Review")

    # Example buttons
    ex1, ex2, ex3 = st.columns(3)
    example_text = ""
    with ex1:
        with st.container():
            if st.button("😊 Positive example"):
                example_text = "This film was absolutely brilliant! The performances were outstanding and the story kept me completely engaged throughout. One of the best movies I have seen in years. The direction was masterful and the cinematography was breathtaking."
    with ex2:
        if st.button("😠 Negative example"):
            example_text = "Terrible movie. The plot made absolutely no sense, the acting was wooden and unconvincing, and the ending was a complete disaster. Total waste of time and money. I cannot believe how bad this was."
    with ex3:
        if st.button("😐 Mixed example"):
            example_text = "The visuals were stunning but the story was painfully slow in the middle. Good performances overall but not quite what I expected. Has its moments but ultimately disappoints."

    user_input = st.text_area(
        label="Review Input",
        value=example_text,
        placeholder="Type or paste a movie review here...",
        height=150,
        label_visibility="collapsed"
    )

    analyse = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)

    # ── Prediction ────────────────────────────────────────
    if analyse:
        if not user_input.strip():
            st.warning("Please enter a review first.")
        else:
            clean  = preprocess(user_input)
            vec    = vectorizer.transform([clean])
            pred   = model.predict(vec)[0]
            proba  = model.predict_proba(vec)[0]
            conf   = max(proba) * 100

            # Result box
            if pred == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div class="result-label" style="color:#4ade80">😊 POSITIVE</div>
                    <div class="result-conf" style="color:#86efac">Confidence: {conf:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div class="result-label" style="color:#f87171">😠 NEGATIVE</div>
                    <div class="result-conf" style="color:#fca5a5">Confidence: {conf:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Probability bars
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.metric("😊 Positive Probability", f"{proba[1]*100:.1f}%")
                st.progress(float(proba[1]))
            with col_neg:
                st.metric("😠 Negative Probability", f"{proba[0]*100:.1f}%")
                st.progress(float(proba[0]))

            # Influential words
            st.markdown("<br>**Key words that influenced this prediction:**", unsafe_allow_html=True)
            feature_names = vectorizer.get_feature_names_out()
            coef          = model.coef_[0]
            vec_array     = vec.toarray()[0]
            nonzero_idx   = vec_array.nonzero()[0]

            if len(nonzero_idx) > 0:
                word_scores = [(feature_names[i], coef[i] * vec_array[i])
                               for i in nonzero_idx]
                word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                top_words = word_scores[:10]

                chips_html = ""
                for word, score in top_words:
                    if score > 0:
                        chips_html += f'<span class="word-chip-pos">▲ {word} ({score:+.2f})</span>'
                    else:
                        chips_html += f'<span class="word-chip-neg">▼ {word} ({score:+.2f})</span>'
                st.markdown(chips_html, unsafe_allow_html=True)

                st.caption("🟢 Green = pushes towards Positive  |  🔴 Red = pushes towards Negative")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── About section ─────────────────────────────────────
    with st.expander("ℹ️ About this Model"):
        st.markdown("""
        **Trained on:** IMDB Movie Reviews dataset (12,000 train / 5,000 test)

        **Preprocessing steps:**
        - Lowercasing, punctuation removal, stopword filtering
        - TF-IDF vectorization (15,000 features, unigrams + bigrams)

        **Model:** Logistic Regression (C=1.0, max_iter=1000)

        **Performance:**
        - Test Accuracy: **86.68%**
        - TF-IDF outperformed Word2Vec by 6.66% on this dataset

        **Deployment:** Streamlit · Practical Activity 1 — DL Algorithm Deployment
        """)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <br>
    Ananya Singh · PRN: 202301100050 · 
</div>
""", unsafe_allow_html=True)