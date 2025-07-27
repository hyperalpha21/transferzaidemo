import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import tempfile

# =======================
# PAGE CONFIG + CSS
# =======================
st.set_page_config(page_title="TransferzAI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
/* GLOBAL THEME - ALL NAVY */
html, body, .stApp, .block-container {
    background-color: #0d1b2a !important; /* deep navy blue */
    color: #f0f4f8 !important; /* light text */
}

/* Remove any default white padding blocks */
.css-18e3th9, .css-1d391kg, .main, .st-emotion-cache-1v0mbdj {
    background-color: transparent !important;
    color: #f0f4f8 !important;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #f0f4f8 !important;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}

/* Main title gradient */
.main-header {
    font-size: 3rem;
    text-align: center;
    font-weight: 700;
    background: linear-gradient(135deg, #82aaff 0%, #b3c7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

/* Section headers */
.step-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: #f0f4f8;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 0.5rem;
}

/* Cards now transparent */
.modern-card {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Buttons with accent gradient */
button[kind="primary"] {
    background: linear-gradient(135deg, #4e5ecf, #667eea);
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Sidebar also navy */
section[data-testid="stSidebar"] {
    background-color: #0a1625 !important;
    color: #f0f4f8 !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Metrics */
.metric-number {
    font-size: 2rem;
    font-weight: 700;
    color: #82aaff;
}

/* Divider lines */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 1rem 0;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05);
    color: #f0f4f8 !important;
}

/* Inputs text area and boxes */
textarea, input, select {
    background-color: rgba(255,255,255,0.05) !important;
    color: #f0f4f8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

/* Table styling */
.dataframe {
    background: rgba(255,255,255,0.03);
    color: #f0f4f8;
}
</style>
""", unsafe_allow_html=True)

# =======================
# STATE DEFAULTS
# =======================
def init_state():
    defaults = {
        'model': None,
        'courses_df': None,
        'courses_emb': None,
        'matches': {},
        'external_courses': []
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =======================
# HELPER FUNCTIONS
# =======================
def extract_level(code: str):
    if not code: return None
    m = re.search(r"(\d{3,4})", str(code))
    if not m: return None
    n = int(m.group(1))
    return 100 if n<200 else 200 if n<300 else 300 if n<400 else 400

def level_bonus(orig, target):
    if orig is None or target is None: return 0.0
    d = abs(orig - target)
    return 0.15 if d==0 else 0.12 if d==100 else 0.02 if d==200 else 0.0

def calculate_transferability_score(t1,d1,t2,d2,model):
    try:
        desc_embs = model.encode([d1,d2])
        sim_desc = cosine_similarity([desc_embs[0]],[desc_embs[1]])[0][0]
        title_embs = model.encode([t1,t2])
        sim_title = cosine_similarity([title_embs[0]],[title_embs[1]])[0][0]
        combined = 1/(1+math.exp(-(-7.144 + 9.219*sim_desc + 5.141*sim_title)))
        return sim_desc,sim_title,combined
    except: return 0,0,0

def get_category(score):
    if score>=0.85: return "Very Likely","🟢"
    elif score>=0.73: return "Likely","🔵"
    elif score>=0.6: return "Needs Review","🟡"
    else: return "Unlikely","🔴"

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['course_title','course_description'])
    df['level'] = df['course_code'].apply(extract_level)
    return df

@st.cache_data
def generate_embeddings(df,_model):
    texts = (df['course_code']+' '+df['course_title']+' '+df['course_description']).tolist()
    arr = _model.encode(texts,show_progress_bar=False)
    return np.array(arr)

def find_matches(external, model, df, embeddings):
    results={}
    for idx,course in enumerate(external):
        title,desc,kw,lvl = course['title'],course['description'],course['keywords'],course['target_level']
        ext_emb = model.encode([title+' '+desc])
        sims = cosine_similarity(ext_emb,embeddings)[0]
        if lvl: sims += df['level'].apply(lambda x:level_bonus(x,lvl)).values
        top_idx = np.argpartition(sims,-5)[-5:]
        sorted_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        matches=[]
        for i in sorted_idx:
            row=df.iloc[i]
            sim_desc,sim_title,score=calculate_transferability_score(
                title,desc,row['course_title'],row['course_description'],model)
            cat,emoji=get_category(score)
            matches.append({
                'code':row['course_code'],
                'title':row['course_title'],
                'score':score,
                'cat':cat,
                'emoji':emoji,
                'sim_desc':sim_desc,
                'sim_title':sim_title
            })
        results[idx]=matches
    return results

# =======================
# UI LAYOUT
# =======================
def main():
    init_state()
    st.markdown('<h1 class="main-header">🎓 TransferzAI</h1>',unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.title("Controls")
        if st.button("Reset"):
            for k in ['model','courses_df','courses_emb','matches','external_courses']:
                st.session_state[k]=None if k not in ['matches','external_courses'] else {} if k=='matches' else []
            st.experimental_rerun()

    # Model loading
    if not st.session_state.model:
        st.write("### 🤖 Load AI Model")
        if st.button("Start AI Model"):
            with st.spinner("Loading model..."):
                st.session_state.model=load_model()
                st.success("✅ Model ready!")
    else:
        st.write("✅ Model Loaded")

    # Catalog loading
    st.markdown('<div class="step-header">📁 Load Course Catalog</div>',unsafe_allow_html=True)
    if st.session_state.model:
        src = st.radio("Data Source",["W&M Catalog","Upload CSV"])
        file="wm_courses_2025.csv" if src=="W&M Catalog" else st.file_uploader("Upload")
        if file and st.button("Load Catalog"):
            df = load_csv(file)
            st.session_state.courses_df=df
            emb = generate_embeddings(df,st.session_state.model)
            st.session_state.courses_emb=emb
            st.success(f"Loaded {len(df)} courses.")

    # Show overview
    if st.session_state.courses_df is not None:
        df=st.session_state.courses_df
        st.write(f"Catalog contains **{len(df)} courses** across **{df['level'].nunique()} levels**")

    # External courses input
    if st.session_state.courses_df is not None:
        st.markdown('<div class="step-header">📚 Add External Courses</div>',unsafe_allow_html=True)
        n=st.slider("Number of external courses",1,5,2)
        external=[]
        for i in range(n):
            with st.expander(f"Course {i+1}"):
                t=st.text_input("Title",key=f"t{i}")
                d=st.text_area("Description",key=f"d{i}")
                k=st.text_input("Keywords",key=f"k{i}")
                l=st.selectbox("Level",[None,100,200,300,400],key=f"l{i}")
                if t and d:
                    external.append({'title':t,'description':d,'keywords':k,'target_level':l})
        if external and st.button("Analyze Courses"):
            matches=find_matches(external,st.session_state.model,
                                 st.session_state.courses_df,
                                 st.session_state.courses_emb)
            st.session_state.external_courses=external
            st.session_state.matches=matches
            st.success("Analysis complete ✅")

    # Show results
    if st.session_state.matches:
        st.markdown('<div class="step-header">🎯 Results</div>',unsafe_allow_html=True)
        counts={'Very Likely':0,'Likely':0,'Needs Review':0,'Unlikely':0}
        for idx,matches in st.session_state.matches.items():
            st.write(f"### External Course {idx+1}: {st.session_state.external_courses[idx]['title']}")
            for rank,m in enumerate(matches,1):
                pct=round(m['score']*100,1)
                counts[m['cat']]+=1
                st.write(f"**#{rank} {m['emoji']} {m['cat']}** – {pct}% → {m['code']}: *{m['title']}*")
                with st.expander("Details"):
                    st.write(f"Description Sim: {m['sim_desc']:.3f}")
                    st.write(f"Title Sim: {m['sim_title']:.3f}")
                    st.write(f"Final Score: {m['score']:.3f}")
            st.markdown("---")

        # Summary counts
        total=sum(counts.values())
        st.write("## 📊 Summary")
        st.write(f"**Total Matches Analyzed:** {total}")
        st.write(f"🟢 Very Likely: {counts['Very Likely']}")
        st.write(f"🔵 Likely: {counts['Likely']}")
        st.write(f"🟡 Needs Review: {counts['Needs Review']}")
        st.write(f"🔴 Unlikely: {counts['Unlikely']}")

if __name__=="__main__":
    main()
