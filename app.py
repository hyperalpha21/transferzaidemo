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

st.markdown('''<style>
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
    font-family: Inter, sans-serif;
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
</style>''', unsafe_allow_html=True)

# =======================
# SESSION STATE DEFAULTS
# =======================
def init_state():
    defaults = {
        "model": None,
        "courses_df": None,
        "courses_emb": None,
        "matches": {},
        "external_courses": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =======================
# HELPERS
# =======================
def extract_level(code: str):
    if not code:
        return None
    try:
        m = re.search(r"(\d{3,4})", code)
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
        score = 1 / (1 + math.exp(-logit))
        return sim_desc, sim_title, score
    except:
        return 0.0, 0.0, 0.0

def classify_score(score):
    if score >= 0.85:
        return "Very Likely", "🟢"
    if score >= 0.7:
        return "Pretty Likely", "🔵"
    if score >= 0.6:
        return "Likely", "🟡"
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
        df = pd.read_csv(path, encoding='latin1')
        df = df.dropna(subset=['course_title', 'course_description'])
        df['course_code'] = df['course_code'].astype(str).str.strip()
        df['level'] = df['course_code'].apply(extract_level)
        return df
    except:
        return None

@st.cache_data
def generate_embeddings(df, _model):
    texts = (df['course_code'] + ' ' + df['course_title'] + ' ' + df['course_description']).tolist()
    return np.array(_model.encode(texts, show_progress_bar=True))

def find_matches(external, model, df, embeddings):
    results = {}
    for idx, course in enumerate(external):
        title, desc, keywords, lvl = course.values()
        ext_text = f"{title} {desc} {keywords or ''}"
        ext_emb = model.encode([ext_text])
        sims = cosine_similarity(ext_emb, embeddings)[0]
        if lvl:
            sims += df['level'].apply(lambda x: level_bonus(x, lvl)).values
        top5 = np.argpartition(sims, -5)[-5:]
        sorted_idx = top5[np.argsort(sims[top5])[::-1]]
        matches = []
        for i in sorted_idx:
            row = df.iloc[i]
            sdesc, stitle, score = calculate_transferability_score(
                title, desc,
                row['course_title'], row['course_description'],
                model
            )
            cat, emoji = classify_score(score)
            matches.append({
                'code': row['course_code'],
                'title': row['course_title'],
                'score': score,
                'cat': cat,
                'emoji': emoji,
                'sim_desc': sdesc,
                'sim_title': stitle,
                'description': row['course_description'][:200] + '...'
            })
        results[idx] = matches
    return results

# =======================
# MAIN UI
# =======================
def main():
    init_state()

    st.markdown('<h1 class="main-header">🎓 TransferzAI</h1>', unsafe_allow_html=True)
    st.markdown('''
    <div class="modern-card">
      <p>We use math and machine learning to estimate course transferability based on your catalog.</p>
      <p>Load the model, upload or use the built‑in catalog, enter external courses, and see match likelihoods!</p>
    </div>
    ''', unsafe_allow_html=True)
    st.info('Disclaimer: 1–2 credit labs may have less certain results.')

    # Sidebar reset
    with st.sidebar:
        st.title('Controls')
        if st.button('Reset App'):
            for k in ['model','courses_df','courses_emb','matches','external_courses']:
                st.session_state[k] = {} if k == 'matches' else None
            st.rerun()

    # Model loader
    if not st.session_state.model:
        st.write('### 🤖 Load AI Model')
        if st.button('Start AI Model'):
            with st.spinner('Loading model…'):
                m = load_model()
                if m:
                    st.session_state.model = m
                    st.success('✅ Model ready!')
                    st.rerun()
    else:
        st.write('✅ **Model Loaded**')

    # Course catalog
    st.markdown('<div class="step-header">📁 Load Course Catalog</div>', unsafe_allow_html=True)
    if st.session_state.model:
        src = st.radio('Source', ['W&M Catalog', 'Upload CSV'])
        file_to_load = None
        if src == 'W&M Catalog' and os.path.exists('wm_courses_2025.csv'):
            file_to_load = 'wm_courses_2025.csv'
        else:
            up = st.file_uploader('Upload Catalog CSV', type='csv')
            if up:
                file_to_load = up
        if file_to_load and st.button('Load Catalog'):
            with st.spinner('Processing catalog…'):
                df = load_csv(file_to_load)
                if df is not None:
                    st.session_state.courses_df = df
                    emb = generate_embeddings(df, st.session_state.model)
                    st.session_state.courses_emb = emb
                    st.success(f'✅ Loaded {len(df)} courses')
    else:
        st.info('Please start AI model first')

    # Catalog preview
    if st.session_state.courses_df is not None:
        df = st.session_state.courses_df
        c1, c2, c3 = st.columns(3)
        c1.metric('Total Courses', len(df))
        c2.metric('Levels', df['level'].nunique())
        c3.metric('Embeddings Ready', 'Yes' if st.session_state.courses_emb is not None else 'No')
        with st.expander('Preview Catalog'):
            st.dataframe(df[['course_code','course_title','level']].head())

    # External courses & analysis
    if st.session_state.courses_df and st.session_state.courses_emb is not None:
        st.markdown('<div class="step-header">📚 Add External Courses</div>', unsafe_allow_html=True)
        n = st.slider('How many external courses?', 1, 15, 2)
        external = []
        for i in range(n):
            with st.expander(f'External Course {i+1}', expanded=(i==0)):
                c1, c2 = st.columns(2)
                with c1:
                    title = st.text_input('Title (required)', key=f'title_{i}',
                                          placeholder='e.g., Intro to Psychology')
                    desc  = st.text_area('Description (required)', key=f'desc_{i}', height=100,
                                         placeholder='Detailed course description…')
                    keywords = st.text_input('Keywords (optional)', key=f'keywords_{i}')
                with c2:
                    lvl_choice = st.selectbox('Target Level',
                                              ['Any Level','100-level','200-level','300-level','400-level'],
                                              index=0, key=f'level_{i}')
                    if lvl_choice == 'Any Level':
                        lvl = None
                    else:
                        lvl = int(lvl_choice.split('-')[0])
                if title and desc:
                    external.append({
                        'title': title,
                        'description': desc,
                        'keywords': keywords,
                        'target_level': lvl
                    })

        if external and st.button('🔍 Analyze', type='primary'):
            with st.spinner('Analyzing…'):
                matches = find_matches(
                    external,
                    st.session_state.model,
                    st.session_state.courses_df,
                    st.session_state.courses_emb
                )
                st.session_state.external_courses = external
                st.session_state.matches = matches
                st.success('✅ Analysis complete!')
                st.rerun()

    # Show results
    if st.session_state.matches:
        st.markdown('<div class="step-header">🎯 Transfer Results</div>', unsafe_allow_html=True)
        summary = {'Very Likely': 0,'Pretty Likely': 0,'Likely': 0,'Unlikely': 0}

        for idx, match_list in st.session_state.matches.items():
            ext = st.session_state.external_courses[idx]
            st.write(f"## External Course {idx+1}: {ext['title']}")
            best = 0.0
            for rank, m in enumerate(match_list, start=1):
                best = max(best, m['score'])
                label = f"#{rank} {m['emoji']} {m['cat']} – {m['score']*100:.1f}% → {m['code']}: {m['title']}"
                with st.expander(label):
                    a, b = st.columns(2)
                    a.metric('Description Sim', f"{m['sim_desc']:.3f}")
                    a.metric('Title Sim', f"{m['sim_title']:.3f}")
                    b.metric('Final Score', f"{m['score']:.3f}")
                    b.metric('Transferability', m['cat'])
                    st.write('**Catalog Description:**', m['description'])
            if best >= 0.85:
                summary['Very Likely'] += 1
            elif best >= 0.7:
                summary['Pretty Likely'] += 1
            elif best >= 0.6:
                summary['Likely'] += 1
            else:
                summary['Unlikely'] += 1
            st.markdown('---')

        st.write('## 📊 Final Course Transferability Summary')
        total = len(st.session_state.external_courses)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Total Input', total)
        c2.metric('🟢 Very Likely', summary['Very Likely'])
        c3.metric('🔵 Pretty Likely', summary['Pretty Likely'])
        c4.metric('🟡 Likely', summary['Likely'])
        c5.metric('🔴 Unlikely', summary['Unlikely'])

if __name__ == "__main__":
    main()
