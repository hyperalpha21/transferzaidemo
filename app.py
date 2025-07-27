import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math

# =======================
# PAGE CONFIG + CSS
# =======================
st.set_page_config(page_title="TransferzAI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
.step-header { font-size: 1.4rem; color: #2e8b57; margin: 2rem 0; padding: 10px; background: #f0f8f0; border-radius: 8px; }
.summary-card { padding: 20px; border-radius: 8px; text-align: center; }
.summary-number { font-size: 3rem; font-weight: bold; }
.summary-label { font-size: 1.2rem; margin-top: 5px; }
.very-high-card { background-color: #c3e6cb; color: #155724; }
.likely-card { background-color: #d4edda; color: #155724; }
.possible-card { background-color: #fff3cd; color: #856404; }
.unlikely-card { background-color: #ffe0b3; color: #8a6d3b; }
.low-card { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# =======================
# SESSION STATE DEFAULTS
# =======================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'courses_df' not in st.session_state:
    st.session_state.courses_df = None
if 'courses_emb' not in st.session_state:
    st.session_state.courses_emb = None
if 'matches' not in st.session_state:
    st.session_state.matches = {}
if 'show_help' not in st.session_state:
    st.session_state.show_help = True

# =======================
# CORE LOGIC FUNCTIONS
# =======================
def calculate_transferability_score(title_ext, desc_ext, title_match, desc_match, model):
    """Compute description + title similarity and combined logit transferability score"""
    try:
        # encode descriptions
        desc_embs = model.encode([desc_ext, desc_match])
        sim_desc = cosine_similarity([desc_embs[0]], [desc_embs[1]])[0][0]

        # encode titles
        title_embs = model.encode([title_ext, title_match])
        sim_title = cosine_similarity([title_embs[0]], [title_embs[1]])[0][0]

        # multi-feature logistic regression (from your 2nd script)
        combined_score = 1/(1 + math.exp(-(-7.144 + 9.219 * sim_desc + 5.141 * sim_title)))
        return sim_desc, sim_title, combined_score
    except Exception as e:
        st.error(f"Transferability calculation error: {e}")
        return None, None, None

def get_transferability_category(score):
    """Classify transferability score to category + emoji"""
    if score >= 0.85:
        return "Very High Transferability", "🟢"
    elif score >= 0.7279793:
        return "Likely Transferable", "🔵"
    elif score >= 0.6:
        return "Possibly Transferable", "🟡"
    elif score >= 0.4:
        return "Unlikely Transferable", "🟠"
    else:
        return "Very Low Transferability", "🔴"

def extract_level(code: str):
    m = re.search(r"(\d{3,4})", code or "")
    if not m:
        return None
    n = int(m.group(1))
    return 100 if n < 200 else 200 if n < 300 else 300 if n < 400 else 400

def level_bonus(orig, target):
    if orig is None or target is None:
        return 0.0
    d = abs(orig - target)
    return 0.15 if d == 0 else 0.12 if d == 100 else 0.02 if d == 200 else 0.0

# =======================
# DATA LOADING FUNCTIONS
# =======================
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

@st.cache_data
def load_csv_data(source_path: str):
    encodings = ['utf-8','latin-1','cp1252','iso-8859-1']
    df = None
    for enc in encodings:
        try:
            path = 'wm_courses_2025.csv' if source_path == 'url' else source_path
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        st.error('Could not read course catalog.')
        return None
    for col in ('course_code','course_title','course_description'):
        if col not in df.columns:
            st.error(f'Missing column: {col}')
            return None
    df = df.dropna(subset=['course_title','course_description'])
    df['level'] = df['course_code'].apply(extract_level)
    return df

@st.cache_data
def generate_embeddings(df: pd.DataFrame, model):
    texts = (df['course_code'] + ' ' + df['course_title'] + ' ' + df['course_description']).tolist()
    import hashlib
    key = hashlib.md5('|'.join(texts).encode()).hexdigest()
    cache_file = Path(f'emb_{key}.pkl')
    if cache_file.exists():
        try:
            return pickle.load(open(cache_file, 'rb'))
        except:
            pass
    prog = st.progress(0)
    embs = []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        emb = model.encode(chunk, show_progress_bar=False)
        embs.extend(emb)
        prog.progress(min((i + batch)/len(texts), 1.0))
    prog.empty()
    arr = np.array(embs)
    try:
        pickle.dump(arr, open(cache_file, 'wb'))
    except:
        pass
    return arr

# =======================
# FIND MATCHES WITH LOGIT
# =======================
def find_matches_with_logit(external_courses, model, df, embeddings):
    """Find top 5 matches by adjusted similarity, then compute logit-based transferability"""
    results = {}

    for idx, course in enumerate(external_courses):
        title_ext = course['title']
        desc_ext = course['description']
        kw = course['keywords']
        target_level = course['target_level']

        if not title_ext or not desc_ext:
            continue

        # keyword filtering
        sub_df = df.copy()
        sub_emb = embeddings
        if kw:
            mask = df.apply(
                lambda r: any(
                    k.strip().lower() in (r['course_code'] + ' ' + r['course_title'] + ' ' + r['course_description']).lower()
                    for k in kw.split(',')
                ), axis=1
            )
            sub_df, sub_emb = df[mask], embeddings[mask.values]
            if sub_df.empty:
                st.warning(f"No matches for keywords: {kw}")
                continue

        # encode external course (full)
        ext_emb = model.encode([f"{title_ext} {desc_ext}"])
        sims = cosine_similarity(ext_emb, sub_emb)[0]

        # apply level bonus
        if target_level:
            sims += sub_df['level'].apply(lambda x: level_bonus(x, target_level)).values

        # get top 5 indices
        top_idx = np.argpartition(sims, -5)[-5:]
        top = top_idx[np.argsort(sims[top_idx])[::-1]]

        matches = []
        for i in top:
            row = sub_df.iloc[i]

            # original vs adjusted sim
            original_sim = cosine_similarity(ext_emb, [sub_emb[i]])[0][0]
            adjusted_sim = sims[i]
            lvl_bonus = adjusted_sim - original_sim

            # run advanced logit model
            sim_desc, sim_title, combined_score = calculate_transferability_score(
                title_ext, desc_ext,
                row['course_title'], row['course_description'],
                model
            )
            category, emoji = get_transferability_category(combined_score)

            matches.append({
                'code': row['course_code'],
                'title': row['course_title'],
                'sim_original': original_sim,
                'sim_adjusted': adjusted_sim,
                'level_bonus': lvl_bonus,
                'sim_desc': sim_desc,
                'sim_title': sim_title,
                'transfer_score': combined_score,
                'category': category,
                'emoji': emoji
            })
        results[idx] = sorted(matches, key=lambda m: m['transfer_score'], reverse=True)
    return results

# =======================
# STREAMLIT MAIN UI
# =======================
def main():
    st.markdown('<h1 class="main-header">🎓 Welcome to TransferzAI</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📋 Menu")
        if st.button("ℹ️ Show/Hide Help"):
            st.session_state.show_help = not st.session_state.show_help
        st.markdown("---")
        if not st.session_state.model:
            if st.button("🚀 Start Model"):
                with st.spinner("Loading model..."):
                    st.session_state.model = load_model()
        else:
            st.success("Model loaded")
        st.markdown("---")
        st.subheader("📁 Course Catalog")
        src = st.radio("Source", ["Upload CSV", "W&M Catalog"], key="src")
        if st.button("🔄 Reset All"):
            for k in ('model', 'courses_df', 'courses_emb', 'matches'):
                st.session_state[k] = None if k != 'matches' else {}
            st.experimental_rerun()

    # Help instructions
    if st.session_state.show_help:
        st.markdown("""
        <div style='background:#f8f9fa;padding:1rem;border-left:4px solid #1f77b4'>
        <h3>How to Use</h3>
        <ol>
            <li>Start the Model in the sidebar</li>
            <li>Load the Course Catalog (upload CSV or use W&M)</li>
            <li>Add your external courses (title, description, optional keywords, level)</li>
            <li>Analyze → It will show top 5 matches + transferability scores</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    # Step 1: Load catalog
    st.markdown('<div class="step-header">📁 Step 1: Load Catalog</div>', unsafe_allow_html=True)
    if src == "Upload CSV":
        file = st.file_uploader("CSV", type="csv")
    else:
        file = "url"; st.info("Using W&M Catalog")
    if file and st.session_state.model and st.button("📂 Load Catalog"):
        df = load_csv_data('url' if src != "Upload CSV" else file)
        if df is not None:
            st.session_state.courses_df = df
            with st.spinner("Embedding courses..."):
                st.session_state.courses_emb = generate_embeddings(df, st.session_state.model)
            st.success("Catalog ready!")

    # Step 2: Add external courses
    external = []
    if st.session_state.courses_df is not None:
        st.markdown('<div class="step-header">📚 Step 2: Add Your Courses</div>', unsafe_allow_html=True)
        n = st.slider("Number of courses", 1, 10, 3)
        for i in range(n):
            with st.expander(f"Course {i+1}", expanded=i < 2):
                t = st.text_input("Title", key=f"t{i}")
                d = st.text_area("Description", key=f"d{i}")
                k = st.text_input("Keywords", key=f"k{i}")
                l = st.selectbox("Level", [None, 100, 200, 300, 400], key=f"l{i}",
                                 format_func=lambda x: "Any" if x is None else f"{x}")
            if t and d:
                external.append({'title': t, 'description': d, 'keywords': k, 'target_level': l})

        # Step 3: Analyze
        if external and st.button("🔍 Analyze Courses"):
            st.session_state.matches = find_matches_with_logit(
                external,
                st.session_state.model,
                st.session_state.courses_df,
                st.session_state.courses_emb
            )

    # Step 4: Results
    if st.session_state.matches:
        st.markdown('<div class="step-header">✅ Results</div>', unsafe_allow_html=True)
        for idx, matches in st.session_state.matches.items():
            st.subheader(f"External Course {idx+1}: {external[idx]['title']}")
            for m in matches:
                pct = round(m['transfer_score'] * 100, 1)
                st.markdown(f"{m['emoji']} **{m['category']}** ({pct}% probability)")
    # show details cleanly
                st.markdown(f"""
            - **{m['code']}**: {m['title']}
                • DescSim: {m['sim_desc']:.3f}  
                • TitleSim: {m['sim_title']:.3f}  
                • AdjustedSim: {m['sim_adjusted']:.3f}
            """)

if __name__ == "__main__":
    main()
