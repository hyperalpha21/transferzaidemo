import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math

# =======================
# PAGE CONFIG + CSS
# =======================
st.set_page_config(page_title="TransferzAI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
html, body, .stApp, .block-container {
    background-color: #0d1b2a !important;
    color: #f0f4f8 !important;
}
.css-18e3th9, .css-1d391kg, .main, .st-emotion-cache-1v0mbdj {
    background-color: transparent !important;
    color: #f0f4f8 !important;
}
h1, h2, h3, h4, h5 {
    color: #f0f4f8 !important;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}
.main-header {
    font-size: 3rem;
    text-align: center;
    font-weight: 700;
    background: linear-gradient(135deg, #82aaff 0%, #b3c7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}
.step-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: #f0f4f8;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 0.5rem;
}
.modern-card {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}
button[kind="primary"] {
    background: linear-gradient(135deg, #4e5ecf, #667eea);
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] {
    background-color: #0a1625 !important;
    color: #f0f4f8 !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}
.metric-number { font-size: 2rem; font-weight: 700; color: #82aaff; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 1rem 0; }
.streamlit-expanderHeader { background: rgba(255,255,255,0.05); color: #f0f4f8 !important; }
textarea, input, select {
    background-color: rgba(255,255,255,0.05) !important;
    color: #f0f4f8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
.dataframe { background: rgba(255,255,255,0.03); color: #f0f4f8; }
</style>
""", unsafe_allow_html=True)

# =======================
# SESSION STATE DEFAULTS
# =======================
def init_state():
    for k, v in {
        "model": None,
        "courses_df": None,
        "courses_emb": None,
        "matches": {},
        "external_courses": []
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =======================
# HELPERS
# =======================
def extract_level(code: str):
    if not code:
        return None
    try:
        m = re.search(r"(\d{3,4})", str(code))
        if not m:
            return None
        n = int(m.group(1))
        return 100 if n < 200 else 200 if n < 300 else 300 if n < 400 else 400
    except:
        return None

def level_bonus(orig, target):
    if orig is None or target is None:
        return 0.0
    d = abs(orig - target)
    return 0.15 if d == 0 else 0.12 if d == 100 else 0.02 if d == 200 else 0.0

def calculate_transferability_score(t1, d1, t2, d2, model):
    try:
        desc_embs = model.encode([d1, d2])
        sim_desc = cosine_similarity([desc_embs[0]], [desc_embs[1]])[0][0]
        title_embs = model.encode([t1, d2])
        sim_title = cosine_similarity([title_embs[0]], [title_embs[1]])[0][0]
        logit = -7.144 + 9.219 * sim_desc + 5.141 * sim_title
        return sim_desc, sim_title, 1 / (1 + math.exp(-logit))
    except:
        return 0, 0, 0

def classify_score(score):
    if score >= 0.7612713:
        return "Very Likely", "🟢"
    elif score >= 0.6649266:
        return "Likely", "🟡"
    elif score >= 0.4340521:
        return "Potentially", "🟠"
    else:
        return "Unlikely", "🔴"

@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except:
        return None

@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path, encoding="latin1")
        df = df.dropna(subset=['course_title', 'course_description'])
        df['course_code'] = df['course_code'].astype(str).str.strip()
        df['level'] = df['course_code'].apply(extract_level)
        return df
    except:
        return None

@st.cache_data
def generate_embeddings(df, _model):
    texts = (df['course_code'] + " " + df['course_title'] + " " + df['course_description']).tolist()
    return np.array(_model.encode(texts, show_progress_bar=True))

def find_matches(external, model, df, embeddings):
    results = {}
    for idx, course in enumerate(external):
        title, desc, kw, lvl = course['title'], course['description'], course['keywords'], course['target_level']
        ext_text = f"{title} {desc} {kw}" if kw else f"{title} {desc}"
        ext_emb = model.encode([ext_text])
        sims = cosine_similarity(ext_emb, embeddings)[0]
        if lvl:
            sims += df['level'].apply(lambda x: level_bonus(x, lvl)).values
        top_idx = np.argpartition(sims, -5)[-5:]
        sorted_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        matches = []
        for i in sorted_idx:
            row = df.iloc[i]
            sdesc, stitle, score = calculate_transferability_score(title, desc, row['course_title'], row['course_description'], model)
            cat, emoji = classify_score(score)
            matches.append({
                "code": row['course_code'],
                "title": row['course_title'],
                "score": score,
                "cat": cat,
                "emoji": emoji,
                "sim_desc": sdesc,
                "sim_title": stitle,
                "description": row['course_description'][:200] + "..."
            })
        results[idx] = matches
    return results

# =======================
# UI
# =======================
def main():
    init_state()
    st.markdown('<h1 class="main-header">🎓 TransferzAI</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class='modern-card'>
    <p>We use math and machine learning to estimate the transferability of a prospective student's courses based on their previous courses taken.</p>
    <p>Simply load the model, upload a CSV file of your courses or use the built-in W&M catalog, then enter all student course titles and descriptions. You will be shown how many of the classes the student is bringing in are likely to transfer!</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Disclaimer: TransferzAI currently focuses primarily on matching **3–4 credit courses** (like lectures). 1–2 credit courses such as labs may have less certain results.")

    with st.sidebar:
        st.title("Controls")
        if st.button("Reset App"):
            for k in ["model", "courses_df", "courses_emb", "matches", "external_courses"]:
                st.session_state[k] = None if k != "matches" else {}
            st.rerun()

    if not st.session_state.model:
        st.write("### 🤖 Load AI Model")
        if st.button("Start AI Model"):
            with st.spinner("Loading model..."):
                m = load_model()
                if m:
                    st.session_state.model = m
                    st.success("✅ Model ready!")
                    st.rerun()
    else:
        st.write("✅ **Model Loaded**")

    st.markdown('<div class="step-header">📁 Load Course Catalog</div>', unsafe_allow_html=True)
    if st.session_state.model:
        src = st.radio("Source", ["W&M Catalog", "Upload CSV"])
        file_to_load = None
        if src == "W&M Catalog" and os.path.exists("wm_courses_2025.csv"):
            file_to_load = "wm_courses_2025.csv"
        else:
            uploaded = st.file_uploader("Upload Catalog CSV", type="csv")
            if uploaded:
                file_to_load = uploaded
        if file_to_load and st.button("Load Catalog"):
            with st.spinner("Processing catalog..."):
                df = load_csv(file_to_load)
                if df is not None:
                    st.session_state.courses_df = df
                    emb = generate_embeddings(df, st.session_state.model)
                    if emb is not None:
                        st.session_state.courses_emb = emb
                        st.success(f"✅ Loaded {len(df)} courses")
    else:
        st.info("Please start AI model first")

    if st.session_state.courses_df is not None:
        df = st.session_state.courses_df
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Courses", len(df))
        col2.metric("Levels", df['level'].nunique())
        col3.metric("Embeddings Ready", "Yes" if st.session_state.courses_emb is not None else "No")
        with st.expander("Preview Catalog"):
            st.dataframe(df[['course_code', 'course_title', 'level']].head())

    if st.session_state.courses_df is not None and st.session_state.courses_emb is not None:
        st.markdown('<div class="step-header">📚 Add External Courses</div>', unsafe_allow_html=True)
        n = st.slider("How many external courses?", 1, 15, 2)
        external = []
        for i in range(n):
            with st.expander(f"External Course {i+1}", expanded=(i == 0)):
                c1, c2 = st.columns(2)
                with c1:
                    t = st.text_input("Title (required)", key=f"t{i}", placeholder="e.g., Introduction to Psychology")
                    d = st.text_area("Description (required)", key=f"d{i}", height=100, placeholder="Detailed course description...")
                with c2:
                    k = st.text_input("Keywords (optional)", key=f"k{i}", placeholder="Additional keywords...")
                    level_choice = st.selectbox(
                        "Target Level",
                        ["Any Level", "100-level", "200-level", "300-level", "400-level"],
                        index=0,
                        key=f"l{i}",
                        help="Course level for better matching"
                    )
                    if not level_choice or level_choice == "Any Level":
                        l = None
                    elif isinstance(level_choice, str) and "-" in level_choice:
                        l = int(level_choice.split("-")[0])
                    else:
                        l = None
                if t and d:
                    external.append({'title': t, 'description': d, 'keywords': k, 'target_level': l})
        if external and st.button("🔍 Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                matches = find_matches(external, st.session_state.model, st.session_state.courses_df, st.session_state.courses_emb)
                st.session_state.external_courses = external
                st.session_state.matches = matches
                st.success("✅ Analysis complete!")
                st.rerun()

    if st.session_state.matches:
        st.markdown('<div class="step-header">🎯 Transfer Results</div>', unsafe_allow_html=True)
        summary_counts = {"Very Likely": 0, "Likely": 0, "Potentially": 0, "Unlikely": 0}

        for idx, matches in st.session_state.matches.items():
            ext_course = st.session_state.external_courses[idx]
            st.write(f"## External Course {idx+1}: {ext_course['title']}")
            best_score = 0
            for rank, m in enumerate(matches, 1):
                if m['score'] > best_score:
                    best_score = m['score']
                with st.expander(f"#{rank} {m['emoji']} {m['cat']} – {round(m['score']*100,1)}% → {m['code']}: {m['title']}"):
                    c1, c2 = st.columns(2)
                    c1.metric("Description Sim", f"{m['sim_desc']:.3f}")
                    c1.metric("Title Sim", f"{m['sim_title']:.3f}")
                    c2.metric("Final Score", f"{m['score']:.3f}")
                    c2.metric("Transferability", m['cat'])
                    st.write("**Catalog Description:**", m['description'])
            cat, _ = classify_score(best_score)
            summary_counts[cat] += 1
            st.markdown("---")

        st.write("## 📊 Final Course Transferability Summary")
        total = len(st.session_state.external_courses)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Input", total)
        c2.metric("🟢 Very Likely", summary_counts["Very Likely"])
        c3.metric("🟡 Likely", summary_counts["Likely"])
        c4.metric("🟠 Potentially", summary_counts["Potentially"])
        c5.metric("🔴 Unlikely", summary_counts["Unlikely"])

if __name__ == "__main__":
    main()
