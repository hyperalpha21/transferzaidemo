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
# PAGE CONFIG & GLOBAL CSS
# =======================
st.set_page_config(page_title="TransferzAI", page_icon="🎓", layout="wide")

# === Minimal Gradient UI (No White Background) ===
st.markdown("""
<style>
/* Soft gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #eef2ff 0%, #f5f7ff 50%, #f7faff 100%);
    color: #1a1a1a;
}

/* Header gradient text */
.main-header {
    font-size: 3.2rem;
    text-align: center;
    font-weight: 700;
    background: linear-gradient(135deg, #4e6af3, #8a3ef3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.8rem;
}

/* Section headers */
.step-header {
    font-size: 1.7rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem;
    color: #333;
}

/* Thin dividers between sections */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));
    margin: 20px 0;
}

/* Help container (transparent) */
.help-container {
    padding-left: 15px;
    border-left: 3px solid #667eea;
}
.help-container h3 { font-size: 1.3rem; margin-bottom: 10px; }
.help-container ol li { padding: 3px 0; }

/* Transfer result box */
.transfer-result {
    backdrop-filter: blur(6px);
    border: 1px solid rgba(102,126,234,0.2);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

/* Metric summary */
.metric-summary {
    padding: 12px;
    border-radius: 8px;
    font-weight: 600;
    text-align: center;
}
.metric-summary.good { color: #28a745; }
.metric-summary.review { color: #ffc107; }
.metric-summary.bad { color: #dc3545; }

</style>
""", unsafe_allow_html=True)

# =======================
# SESSION STATE
# =======================
def init_session_state():
    defaults = {
        'model': None,
        'courses_df': None,
        'courses_emb': None,
        'matches': {},
        'external_courses': [],
        'show_help': True
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =======================
# CORE LOGIC
# =======================
def calculate_transferability_score(title_ext, desc_ext, title_match, desc_match, model):
    """Compute similarity & logit transferability probability"""
    try:
        desc_embs = model.encode([desc_ext or "", desc_match or ""])
        sim_desc = cosine_similarity([desc_embs[0]], [desc_embs[1]])[0][0]
        title_embs = model.encode([title_ext or "", title_match or ""])
        sim_title = cosine_similarity([title_embs[0]], [title_embs[1]])[0][0]
        combined = 1/(1+math.exp(-(-7.144 + 9.219*sim_desc + 5.141*sim_title)))
        return sim_desc, sim_title, combined
    except:
        return 0.0, 0.0, 0.0

def get_category(score):
    if score >= 0.85: return "Very High Transferability","🟢"
    elif score >= 0.7279793: return "Likely Transferable","🔵"
    elif score >= 0.6: return "Possibly Transferable","🟡"
    elif score >= 0.4: return "Unlikely Transferable","🟠"
    else: return "Very Low Transferability","🔴"

def extract_level(code):
    if not code: return None
    m=re.search(r"(\d{3,4})",str(code))
    if not m: return None
    n=int(m.group(1))
    return 100 if n<200 else 200 if n<300 else 300 if n<400 else 400

def level_bonus(orig,target):
    if orig is None or target is None: return 0.0
    d=abs(orig-target)
    return 0.15 if d==0 else 0.12 if d==100 else 0.02 if d==200 else 0.0

# =======================
# DATA LOADING
# =======================
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except: return None

@st.cache_data
def load_csv_data(src):
    encs=['utf-8','latin-1','cp1252']
    df=None
    if src=='url':
        for e in encs:
            try:
                df=pd.read_csv('wm_courses_2025.csv',encoding=e)
                break
            except: continue
    elif hasattr(src,'read'):
        for e in encs:
            try:
                src.seek(0)
                df=pd.read_csv(src,encoding=e)
                break
            except: continue
    if df is None: return None
    for c in ['course_code','course_title','course_description']:
        if c not in df.columns: return None
    df=df.dropna(subset=['course_title','course_description'])
    df['level']=df['course_code'].apply(extract_level)
    return df

@st.cache_data
def generate_embeddings(df,model):
    if model is None: return None
    texts=(df['course_code']+' '+df['course_title']+' '+df['course_description']).tolist()
    import hashlib
    key=hashlib.md5('|'.join(texts).encode()).hexdigest()
    cache_file=Path(tempfile.gettempdir())/f'emb_{key}.pkl'
    if cache_file.exists():
        try:
            with open(cache_file,'rb') as f:return pickle.load(f)
        except:pass
    embs=[]; prog=st.progress(0)
    for i in range(0,len(texts),16):
        chunk=texts[i:i+16]
        embs.extend(model.encode(chunk,show_progress_bar=False))
        prog.progress(min((i+16)/len(texts),1.0))
    prog.empty()
    arr=np.array(embs)
    try:
        with open(cache_file,'wb') as f:pickle.dump(arr,f)
    except:pass
    return arr

# =======================
# MATCH FINDING
# =======================
def find_matches(external,model,df,embs):
    if model is None or df is None or embs is None: return {}
    results={}
    for idx,course in enumerate(external):
        title=course['title']; desc=course['description']
        kws=course['keywords']; tgt=course['target_level']
        if not title.strip() or not desc.strip(): continue
        sub_df=df.copy(); sub_emb=embs.copy()
        if kws.strip():
            kwlist=[k.strip().lower() for k in kws.split(',')]
            mask=df.apply(lambda r:any(k in (str(r['course_code'])+' '+str(r['course_title'])+' '+str(r['course_description'])).lower() for k in kwlist),axis=1)
            if mask.any(): sub_df=df[mask].copy(); sub_emb=embs[mask.values]
            else: continue
        if sub_df.empty: continue
        ext_text=f"{title} {desc}"
        ext_emb=model.encode([ext_text])
        sims=cosine_similarity(ext_emb,sub_emb)[0]
        if tgt: sims+=sub_df['level'].apply(lambda x:level_bonus(x,tgt)).to_numpy()
        top_idx=np.argpartition(sims,-min(5,len(sims)))[-min(5,len(sims)):]
        top_sorted=top_idx[np.argsort(sims[top_idx])[::-1]]
        matches=[]
        for i in top_sorted:
            row=sub_df.iloc[i]
            sim_desc,sim_title,score=calculate_transferability_score(title,desc,row['course_title'],row['course_description'],model)
            cat,emoji=get_category(score)
            matches.append({'code':row['course_code'],'title':row['course_title'],
                            'transfer_score':score,'emoji':emoji,'category':cat})
        results[idx]=sorted(matches,key=lambda m:m['transfer_score'],reverse=True)
    return results

# =======================
# MAIN UI
# =======================
def main():
    init_session_state()
    st.markdown('<h1 class="main-header">🎓 TransferzAI</h1>',unsafe_allow_html=True)

    # Help
    if st.session_state.show_help:
        st.markdown("""
        <div class='help-container'>
        <h3>How it works</h3>
        <ol>
        <li>Load the AI model</li>
        <li>Upload/Load course catalog</li>
        <li>Add external courses</li>
        <li>Run AI to find transferability</li>
        </ol>
        </div>
        <hr class="section-divider">
        """,unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        if st.button("Toggle Help"): st.session_state.show_help=not st.session_state.show_help
        if st.button("Reset App"):
            for k in ['model','courses_df','courses_emb','matches','external_courses']:
                st.session_state[k]=None if k not in ['matches','external_courses'] else {} if k=='matches' else []
            st.rerun()

    # Model setup
    if not st.session_state.model:
        st.subheader("🤖 Load AI Model")
        if st.button("Start AI Model"):
            with st.spinner("Loading..."):
                st.session_state.model=load_model()
                if st.session_state.model: st.success("✅ Model Ready!"); st.balloons()
                else: st.error("❌ Failed.")
    else:
        st.markdown("✅ **Model Loaded**")

    # Step 1: Catalog
    st.markdown('<h2 class="step-header">📁 Load Course Catalog</h2>',unsafe_allow_html=True)
    src=st.radio("Catalog Source",["Upload CSV","Use Built-in"],horizontal=True)
    file=None
    if src=="Upload CSV": file=st.file_uploader("Upload CSV",type="csv")
    else: file="url"; st.info("Using built-in catalog")
    if file and st.session_state.model:
        if st.button("Load Catalog"):
            with st.spinner("Loading catalog..."):
                df=load_csv_data(file)
                if df is not None:
                    st.session_state.courses_df=df
                    st.success(f"Loaded {len(df)} courses")
                    with st.spinner("Generating embeddings..."):
                        e=generate_embeddings(df,st.session_state.model)
                        if e is not None: st.session_state.courses_emb=e; st.success("Embeddings Ready!")
                        else: st.error("Embedding failed")

    # Catalog info
    if st.session_state.courses_df is not None:
        with st.expander("Catalog Overview"):
            df=st.session_state.courses_df
            st.write(f"**Total Courses:** {len(df)} | Levels: {df['level'].nunique()}")
            st.dataframe(df[['course_code','course_title']].head())

    # Step 2: External
    if st.session_state.courses_df is not None and st.session_state.model is not None:
        st.markdown('<h2 class="step-header">📚 Add External Courses</h2>',unsafe_allow_html=True)
        n=st.slider("How many?",1,5,2)
        ext=[]
        for i in range(n):
            with st.expander(f"External Course {i+1}",expanded=(i<2)):
                t=st.text_input("Title",key=f"t{i}")
                d=st.text_area("Description",key=f"d{i}")
                k=st.text_input("Keywords",key=f"k{i}")
                lvl=st.selectbox("Target Level",[None,100,200,300,400],key=f"lvl{i}",
                                 format_func=lambda x:"Any" if x is None else f"{x}-level")
                if t and d: ext.append({'title':t,'description':d,'keywords':k,'target_level':lvl})
        st.session_state.external_courses=ext

        if ext:
            st.markdown('<h2 class="step-header">🔍 Run Analysis</h2>',unsafe_allow_html=True)
            if st.button("Start Analysis"):
                with st.spinner("Analyzing..."):
                    m=find_matches(ext,st.session_state.model,st.session_state.courses_df,st.session_state.courses_emb)
                    st.session_state.matches=m
                    if m: st.success("Done!"); st.balloons()
                    else: st.warning("No matches.")

    # Step 3: Results
    if st.session_state.matches:
        st.markdown('<h2 class="step-header">🎯 Results</h2>',unsafe_allow_html=True)
        total_good=total_review=total_bad=0
        for idx,matches in st.session_state.matches.items():
            ext=st.session_state.external_courses[idx]
            st.subheader(f"📘 {ext['title']}")
            if not matches: st.warning("No matches"); continue
            top=matches[0]
            pct=round(top['transfer_score']*100,1)
            st.markdown(f"""
            <div class="transfer-result">
            <h4>{top['emoji']} {top['category']}</h4>
            <p><strong>{pct}%</strong> chance to transfer as **{top['code']} – {top['title']}**</p>
            </div>
            """,unsafe_allow_html=True)
            # Count category for summary
            if top['category'] in ["Very High Transferability","Likely Transferable"]: total_good+=1
            elif top['category']=="Possibly Transferable": total_review+=1
            else: total_bad+=1
        st.markdown("<hr class='section-divider'>",unsafe_allow_html=True)
        st.subheader("📊 Summary")
        st.markdown(f"<div class='metric-summary good'>✅ Likely Transfer: {total_good}</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-summary review'>🟡 Need Review: {total_review}</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-summary bad'>❌ Unlikely Transfer: {total_bad}</div>",unsafe_allow_html=True)

if __name__=="__main__":
    main()
