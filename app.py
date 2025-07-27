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
# PAGE CONFIG + MODERN CSS
# =======================
st.set_page_config(page_title="TransferzAI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
body { font-family: 'Inter', sans-serif; }
.main-header {
    font-size: 3.2rem;
    text-align: center;
    font-weight: 600;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.step-header {
    font-size: 1.8rem;
    font-weight: 600;
    margin: 2.5rem 0 1rem 0;
}
.modern-card {
    background: #fff;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}
.modern-card:hover { transform: translateY(-2px); }
.help-container {
    background: #f6f8ff;
    padding: 24px;
    border-radius: 14px;
    border-left: 4px solid #667eea;
    margin-bottom: 22px;
}
.status-badge {
    background: #e7f9ef;
    color: #198754;
    padding: 6px 14px;
    border-radius: 20px;
    display: inline-block;
    font-weight: 600;
}
.metric-display {
    background: #f8f9ff;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    border: 1px solid #e6e8ff;
    margin-bottom: 12px;
}
.metric-number {
    font-size: 2.2rem;
    font-weight: 700;
    color: #667eea;
}
.metric-label {
    font-size: 13px;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.transfer-result {
    background: #fff;
    border: 1px solid #eee;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.percentage-text {
    font-weight: 700;
    color: #667eea;
}
.code-tag {
    background: #f1f3f8;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =======================
# SESSION DEFAULTS
# =======================
def init_state():
    defaults = {
        "model": None,
        "courses_df": None,
        "courses_emb": None,
        "matches": {},
        "external_courses": [],
        "show_help": True
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =======================
# CORE MATH + LOGIC
# =======================
def extract_level(code):
    if not code: return None
    m = re.search(r"(\d{3,4})", str(code))
    if not m: return None
    n = int(m.group(1))
    return 100 if n<200 else 200 if n<300 else 300 if n<400 else 400

def level_bonus(orig,target):
    if orig is None or target is None: return 0.0
    d = abs(orig - target)
    return 0.15 if d==0 else 0.12 if d==100 else 0.02 if d==200 else 0.0

def calculate_transferability_score(title_ext,desc_ext,title_match,desc_match,model):
    try:
        desc_embs = model.encode([desc_ext,desc_match])
        sim_desc = cosine_similarity([desc_embs[0]],[desc_embs[1]])[0][0]
        title_embs = model.encode([title_ext,title_match])
        sim_title = cosine_similarity([title_embs[0]],[title_embs[1]])[0][0]
        score = 1/(1+math.exp(-(-7.144 + 9.219*sim_desc + 5.141*sim_title)))
        return sim_desc, sim_title, score
    except:
        return 0,0,0

def get_transferability_category(score):
    if score>=0.85: return "Very High Transferability","🟢"
    elif score>=0.7279: return "Likely Transferable","🔵"
    elif score>=0.6: return "Possibly Transferable","🟡"
    elif score>=0.4: return "Unlikely Transferable","🟠"
    else: return "Very Low Transferability","🔴"

# =======================
# DATA LOAD + EMBEDDINGS
# =======================
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_data
def load_csv_data(src):
    encs=['utf-8','latin-1','cp1252']
    df=None
    path='wm_courses_2025.csv' if src=='url' else src
    for enc in encs:
        try:
            df=pd.read_csv(path,encoding=enc)
            break
        except: continue
    if df is None: return None
    needed=['course_code','course_title','course_description']
    for col in needed:
        if col not in df.columns: return None
    df=df.dropna(subset=['course_title','course_description'])
    df['level']=df['course_code'].apply(extract_level)
    return df

@st.cache_data
def generate_embeddings(df,_model):
    texts=(df['course_code']+' '+df['course_title']+' '+df['course_description']).tolist()
    cache=Path(tempfile.gettempdir())/f"emb_{hash('|'.join(texts))}.pkl"
    if cache.exists(): return pickle.load(open(cache,'rb'))
    embs=_model.encode(texts,show_progress_bar=True)
    pickle.dump(embs,open(cache,'wb'))
    return np.array(embs)

# =======================
# FIND MATCHES
# =======================
def find_matches_with_logit(external_courses,model,df,embeddings):
    results={}
    for idx,course in enumerate(external_courses):
        title_ext,desc_ext=course['title'],course['description']
        kw=course['keywords']
        target_level=course['target_level']
        sub_df=df; sub_emb=embeddings
        if kw:
            keys=[k.strip().lower() for k in kw.split(',')]
            mask=df.apply(lambda r:any(k in (r['course_code']+r['course_title']+r['course_description']).lower() for k in keys),axis=1)
            if not mask.any(): continue
            sub_df=df[mask]; sub_emb=embeddings[mask.values]
        ext_emb=model.encode([f"{title_ext} {desc_ext}"])
        sims=cosine_similarity(ext_emb,sub_emb)[0]
        if target_level:
            sims+=sub_df['level'].apply(lambda x:level_bonus(x,target_level)).to_numpy()
        top_idx=np.argsort(sims)[-5:][::-1]
        matches=[]
        for i in top_idx:
            row=sub_df.iloc[i]
            orig_sim=cosine_similarity(ext_emb,[sub_emb[i]])[0][0]
            adj_sim=sims[i]
            sim_desc,sim_title,score=calculate_transferability_score(
                title_ext,desc_ext,row['course_title'],row['course_description'],model
            )
            cat,emoji=get_transferability_category(score)
            matches.append({
                "code":row['course_code'],"title":row['course_title'],
                "sim_original":orig_sim,"sim_adjusted":adj_sim,
                "sim_desc":sim_desc,"sim_title":sim_title,
                "transfer_score":score,"category":cat,"emoji":emoji
            })
        results[idx]=sorted(matches,key=lambda m:m['transfer_score'],reverse=True)
    return results

# =======================
# MAIN UI
# =======================
def main():
    init_state()
    st.markdown('<h1 class="main-header">🎓 TransferzAI</h1>',unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.title("⚙️ Controls")
        if st.button("ℹ️ Toggle Help"): st.session_state.show_help=not st.session_state.show_help
        if st.button("🔄 Reset All"):
            for k in ['model','courses_df','courses_emb','matches','external_courses']:
                st.session_state[k] = None if k!="matches" else {}
            st.rerun()

    # HELP
    if st.session_state.show_help:
        st.markdown("""
        <div class="help-container">
        <b>How it works:</b><br>
        1️⃣ Load AI model<br>
        2️⃣ Load Course Catalog<br>
        3️⃣ Add External Courses<br>
        4️⃣ Run AI Analysis → See transferability probabilities
        </div>
        """,unsafe_allow_html=True)

    # STEP 1: LOAD MODEL
    if not st.session_state.model:
        st.markdown('<div class="modern-card">',unsafe_allow_html=True)
        st.subheader("🤖 Load AI Model")
        st.write("Start the AI model to enable course analysis")
        if st.button("Start AI Model"):
            with st.spinner("Loading model..."):
                st.session_state.model=load_model()
            if st.session_state.model: st.success("✅ Model Loaded!")
        st.markdown('</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge">✅ AI Model Ready</div>',unsafe_allow_html=True)

    # STEP 2: LOAD CATALOG
    if st.session_state.model:
        st.markdown('<div class="step-header">📁 Load Course Catalog</div>',unsafe_allow_html=True)
        st.markdown('<div class="modern-card">',unsafe_allow_html=True)
        src=st.radio("Catalog Source",["Upload CSV","Use W&M Catalog"],horizontal=True)
        file=st.file_uploader("Upload CSV",type="csv") if src=="Upload CSV" else "url"
        if file and st.button("Load Catalog"):
            df=load_csv_data(file)
            if df is not None:
                st.session_state.courses_df=df
                st.success(f"✅ Loaded {len(df)} courses")
                with st.spinner("Generating embeddings..."):
                    st.session_state.courses_emb=generate_embeddings(df,st.session_state.model)
        st.markdown('</div>',unsafe_allow_html=True)

    # Show Catalog Stats
    if st.session_state.courses_df is not None:
        with st.expander("📊 Catalog Overview"):
            df=st.session_state.courses_df
            col1,col2,col3=st.columns(3)
            with col1:
                st.markdown('<div class="metric-display">',unsafe_allow_html=True)
                st.markdown(f'<div class="metric-number">{len(df)}</div><div class="metric-label">Total Courses</div>',unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-display">',unsafe_allow_html=True)
                depts=df['course_code'].str[:4].nunique()
                st.markdown(f'<div class="metric-number">{depts}</div><div class="metric-label">Departments</div>',unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-display">',unsafe_allow_html=True)
                lvls=df['level'].nunique()
                st.markdown(f'<div class="metric-number">{lvls}</div><div class="metric-label">Course Levels</div>',unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)
            st.dataframe(df[['course_code','course_title']].head(),use_container_width=True)

    # STEP 3: ADD EXTERNAL COURSES
    external=[]
    if st.session_state.courses_df is not None:
        st.markdown('<div class="step-header">📚 Add External Courses</div>',unsafe_allow_html=True)
        st.markdown('<div class="modern-card">',unsafe_allow_html=True)
        num=st.slider("Number of courses to analyze",1,5,2)
        for i in range(num):
            with st.expander(f"📝 Course {i+1}",expanded=(i<2)):
                col1,col2=st.columns([2,1])
                with col1:
                    t=st.text_input("Title",key=f"t{i}")
                    d=st.text_area("Description",key=f"d{i}",height=120)
                with col2:
                    k=st.text_input("Keywords",key=f"k{i}")
                    l=st.selectbox("Target Level",[None,100,200,300,400],key=f"l{i}",format_func=lambda x:"Any" if x is None else f"{x}-level")
                if t and d:
                    external.append({"title":t,"description":d,"keywords":k,"target_level":l})
        st.session_state.external_courses=external
        st.markdown('</div>',unsafe_allow_html=True)

        if external and st.button("Run AI Analysis"):
            with st.spinner("Analyzing transferability..."):
                st.session_state.matches=find_matches_with_logit(
                    external,st.session_state.model,
                    st.session_state.courses_df,
                    st.session_state.courses_emb
                )
            st.success("✅ Analysis complete!")

    # STEP 4: RESULTS
    if st.session_state.matches:
        st.markdown('<div class="step-header">🎯 Transferability Results</div>',unsafe_allow_html=True)
        st.write("*Percentage = probability your external course transfers as this match*")
        for idx,matches in st.session_state.matches.items():
            ext=st.session_state.external_courses[idx]
            st.subheader(f"📘 {ext['title']}")
            if not matches: st.warning("No matches found"); continue
            for rank,m in enumerate(matches,1):
                pct=round(m['transfer_score']*100,1)
                st.markdown(f"""
                <div class="transfer-result">
                <b>#{rank} {m['emoji']} {m['category']}</b> → <span class="percentage-text">{pct}%</span> likely<br>
                <span class="code-tag">{m['code']}</span> {m['title']}
                </div>""",unsafe_allow_html=True)
                with st.expander("📊 Detailed Metrics"):
                    st.write(f"Description Similarity: {m['sim_desc']:.3f}")
                    st.write(f"Title Similarity: {m['sim_title']:.3f}")
                    st.write(f"Base Similarity: {m['sim_original']:.3f}")
                    st.write(f"Adjusted Similarity: {m['sim_adjusted']:.3f}")
            st.markdown("---")

if __name__=="__main__":
    main()
