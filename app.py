import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Similarity threshold for auto-classification
TRANSFER_THRESHOLD = 0.6  

# Page config
st.set_page_config(
    page_title='TransferzAI',
    page_icon='🎓',
    layout='wide'
)

# Custom CSS for styling
st.markdown('''<style>
.main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
.step-header { font-size: 1.4rem; color: #2e8b57; margin: 2rem 0; padding: 10px; background: #f0f8f0; border-radius: 8px; }
.help-text { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 10px 0; color: #333; }
.summary-card { padding: 20px; border-radius: 8px; text-align: center; }
.summary-number { font-size: 3rem; font-weight: bold; }
.summary-label { font-size: 1.2rem; margin-top: 5px; }
.very-likely-card { background-color: #c3e6cb; color: #155724; }
.likely-card { background-color: #d4edda; color: #155724; }
.review-card { background-color: #fff3cd; color: #856404; }
</style>''', unsafe_allow_html=True)

# Session state defaults
if 'model' not in st.session_state: st.session_state.model = None
if 'courses_df' not in st.session_state: st.session_state.courses_df = None
if 'courses_emb' not in st.session_state: st.session_state.courses_emb = None
if 'matches' not in st.session_state: st.session_state.matches = {}
if 'show_help' not in st.session_state: st.session_state.show_help = True

class CourseTransferChecker:
    @st.cache_resource
    def load_model(_self):
        try:
            return SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            st.error(f'Model load error: {e}')
            return None

    def extract_level(self, code: str):
        m = re.search(r"(\d{3,4})", code or "")
        if not m: return None
        n = int(m.group(1))
        return 100 if n < 200 else 200 if n < 300 else 300 if n < 400 else 400

    def level_bonus(self, orig, target):
        if orig is None or target is None: return 0.0
        d = abs(orig - target)
        return 0.15 if d == 0 else 0.12 if d == 100 else 0.02 if d == 200 else 0.0

    @st.cache_data
    def load_csv_data(_self, source_path: str):
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
        df['level'] = df['course_code'].apply(_self.extract_level)
        return df

    @st.cache_data
    def generate_embeddings(_self, df: pd.DataFrame):
        texts = (df['course_code'] + ' ' + df['course_title'] + ' ' + df['course_description']).tolist()
        import hashlib
        key = hashlib.md5('|'.join(texts).encode()).hexdigest()
        cache_file = Path(f'emb_{key}.pkl')
        if cache_file.exists():
            try: return pickle.load(open(cache_file,'rb'))
            except: pass
        model = st.session_state.model
        if not model:
            st.error('Model not loaded.')
            return None
        prog = st.progress(0)
        embs = []
        batch = 16
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            emb = model.encode(chunk, show_progress_bar=False)
            embs.extend(emb)
            prog.progress(min((i+batch)/len(texts), 1.0))
        prog.empty()
        arr = np.array(embs)
        try: pickle.dump(arr, open(cache_file,'wb'))
        except: pass
        return arr

    def find_matches(self, external: list, df: pd.DataFrame, embs: np.ndarray):
        results = {}
        st.info('🔍 Finding matches...')
        for idx, course in enumerate(external):
            title, desc, kw, lvl = course.values()
            sub_df, sub_emb = df, embs
            if kw:
                mask = df.apply(lambda r: any(k.strip().lower() in (r['course_code']+' '+r['course_title']+' '+r['course_description']).lower() for k in kw.split(',')), axis=1)
                sub_df, sub_emb = df[mask], embs[mask.values]
                if sub_df.empty:
                    st.warning(f'No matches for keywords: {kw}')
                    continue
            ext_emb = st.session_state.model.encode([f"{title} {desc}"])
            sims = cosine_similarity(ext_emb, sub_emb)[0]
            if lvl:
                sims += sub_df['level'].apply(lambda x: self.level_bonus(x, lvl)).values
            top_idx = np.argpartition(sims, -5)[-5:]
            top = top_idx[np.argsort(sims[top_idx])[::-1]]
            hits = [{'code': sub_df.iloc[i]['course_code'], 'title': sub_df.iloc[i]['course_title'], 'sim': sims[i]} for i in top]
            if hits:
                results[idx] = hits
        return results

# Build UI

def main():
    st.markdown('<h1 class="main-header">🎓 Will My Courses Transfer?</h1>', unsafe_allow_html=True)
    checker = CourseTransferChecker()

    with st.sidebar:
        st.title('📋 Menu')
        if st.button('ℹ️ Show/Hide Help'):
            st.session_state.show_help = not st.session_state.show_help
        st.markdown('---')
        if not st.session_state.model:
            if st.button('🚀 Start Model'):
                with st.spinner('Loading model...'):
                    st.session_state.model = checker.load_model()
        else:
            st.success('Model loaded')
        st.markdown('---')
        st.subheader('📁 Course Catalog')
        src = st.radio('Source', ['Upload CSV', 'Built-in catalog'], key='src')
        if st.button('🔄 Reset All'):
            for k in ('model', 'courses_df', 'courses_emb', 'matches'):
                st.session_state[k] = None if k != 'matches' else {}
            st.experimental_rerun()

    if st.session_state.show_help:
        st.markdown('<div class="help-text"><h3>How to Use</h3><ol><li>Start model</li><li>Load catalog</li><li>Add courses</li><li>Analyze transfer</li></ol></div>', unsafe_allow_html=True)

    # Step 1: Load catalog
    st.markdown('<div class="step-header">📁 Step 1: Load Catalog</div>', unsafe_allow_html=True)
    if src == 'Upload CSV':
        file = st.file_uploader('CSV', type='csv')
    else:
        file = 'url'; st.info('Using built-in catalog')
    if file and st.session_state.model and st.button('📂 Load Catalog'):
        df = checker.load_csv_data('url' if src != 'Upload CSV' else file)
        if df is not None:
            st.session_state.courses_df = df
            with st.spinner('Embedding courses...'):
                st.session_state.courses_emb = checker.generate_embeddings(df)
            st.success('Catalog ready!')

    # Step 2: Add external courses
    external = []
    if st.session_state.courses_df is not None:
        st.markdown('<div class="step-header">📚 Step 2: Add Your Courses</div>', unsafe_allow_html=True)
        n = st.slider('Number of courses', 1, 10, 3)
        for i in range(n):
            with st.expander(f'Course {i+1}', expanded=i<2):
                t = st.text_input('Title', key=f't{i}')
                d = st.text_area('Description', key=f'd{i}')
                k = st.text_input('Keywords', key=f'к{i}')
                l = st.selectbox('Level', [None, 100, 200, 300, 400], key=f'l{i}', format_func=lambda x: 'Any' if x is None else f'{x}')
            if t and d:
                external.append({'title': t, 'description': d, 'keywords': k, 'target_level': l})

        # Analyze courses
        if external and st.button('🔍 Analyze Courses'):
            st.session_state.matches = checker.find_matches(
                external, st.session_state.courses_df, st.session_state.courses_emb
            )

    # Step 3: Display results
    if st.session_state.matches:
        st.markdown('<div class="step-header">✅ Results</div>', unsafe_allow_html=True)
        total = len(external)
        # categorize each course by top 3 consensus
        counts = {'very_likely': 0, 'likely': 0, 'review': 0}
        course_cats = {}
        for idx, hits in st.session_state.matches.items():
            top3 = hits[:3]
            above = sum(1 for h in top3 if h['sim'] >= TRANSFER_THRESHOLD)
            if above == 3:
                counts['very_likely'] += 1
                course_cats[idx] = 'very_likely'
            elif above >= 1:
                counts['likely'] += 1
                course_cats[idx] = 'likely'
            else:
                counts['review'] += 1
                course_cats[idx] = 'review'

        # Summary cards
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"""
            <div class="summary-card very-likely-card">
                <div class="summary-number">{counts['very_likely']}</div>
                <div class="summary-label">Very Likely</div>
            </div>
        """, unsafe_allow_html=True)
        col2.markdown(f"""
            <div class="summary-card likely-card">
                <div class="summary-number">{counts['likely']}</div>
                <div class="summary-label">Likely</div>
            </div>
        """, unsafe_allow_html=True)
        col3.markdown(f"""
            <div class="summary-card review-card">
                <div class="summary-number">{counts['review']}</div>
                <div class="summary-label">Needs Review</div>
            </div>
        """, unsafe_allow_html=True)

        # Detailed per-course feedback
        for idx, hits in st.session_state.matches.items():
            course = external[idx]['title']
            cat = course_cats[idx]
            best = hits[0]
            pct = round(best['sim'] * 100, 1)
            if cat == 'very_likely':
                st.success(f"✅ {course}: Very Likely ({pct}%)")
            elif cat == 'likely':
                st.info(f"ℹ️ {course}: Likely ({pct}%)")
            else:
                st.warning(f"⚠️ {course}: Needs Review ({pct}%)")
                st.markdown('Top alternatives:')
                for alt in hits[:3]:
                    st.markdown(f"- {alt['code']}: {alt['title']} ({round(alt['sim']*100, 1)}%)")
            st.markdown('---')

if __name__ == '__main__':
    main()
