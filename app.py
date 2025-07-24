import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
import pickle
from pathlib import Path

# Threshold for auto-classifying transferability
TRANSFER_THRESHOLD = 0.75

# Page configuration
st.set_page_config(
    page_title='Welcome to TransferzAI',
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown('''<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.step-header {
    font-size: 1.4rem;
    color: #2e8b57;
    margin-top: 2rem;
    margin-bottom: 2rem;
    padding: 10px;
    background-color: #f0f8f0;
    border-radius: 8px;
}
.help-text {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #17a2b8;
    margin: 10px 0;
    color: #333333;
}
</style>''', unsafe_allow_html=True)

# Initialize session state
for key in ('model','university_courses_df','university_embeddings','course_matches','show_help'):
    if key not in st.session_state:
        st.session_state[key] = {} if key=='course_matches' else (True if key=='show_help' else None)

class CourseTransferChecker:
    @st.cache_resource
    def load_model(_self):
        try:
            return SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            st.error(f'Error loading model: {e}')
            return None

    def extract_course_level(self, code):
        try:
            num = re.search(r'(\d{3,4})', code)
            if not num:
                return None
            level = int(num.group(1))
            if level < 200:
                return 100
            elif level < 300:
                return 200
            elif level < 400:
                return 300
            else:
                return 400
        except:
            return None

    def calculate_level_bonus(self, course_level, target_level):
        if course_level is None or target_level is None:
            return 0.0
        diff = abs(course_level - target_level)
        if diff == 0:
            return 0.15
        elif diff == 100:
            return 0.12
        elif diff == 200:
            return 0.02
        return 0.0

    @st.cache_data
    def load_csv_data(_self, source):
        try:
            encodings = ['utf-8','latin-1','cp1252','iso-8859-1']
            df = None
            for enc in encodings:
                try:
                    path = 'wm_courses_2025.csv' if source=='url' else source
                    df = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                st.error('Could not load CSV; unsupported format.')
                return None
            required = ['course_code','course_title','course_description']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f'Missing required columns: {missing}')
                return None
            df.dropna(subset=['course_title','course_description'], inplace=True)
            df['course_level'] = df['course_code'].apply(_self.extract_course_level)
            return df
        except Exception as e:
            st.error(f'Error loading catalog: {e}')
            return None

    @st.cache_data
    def generate_embeddings(_self, df):
        texts = df['course_code'] + ' ' + df['course_title'] + ' ' + df['course_description']
        import hashlib
        key = hashlib.md5('||'.join(texts).encode()).hexdigest()
        cache_file = Path(f'emb_{key}.pkl')
        if cache_file.exists():
            try:
                return pickle.load(open(cache_file,'rb'))
            except:
                pass
        model = st.session_state.model
        if model is None:
            st.error('Model not loaded.')
            return None
        st.info('Encoding courses...')
        prog = st.progress(0)
        all_emb = []
        batch = 16
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch].tolist()
            emb = model.encode(chunk, show_progress_bar=False)
            all_emb.extend(emb)
            prog.progress(min((i+batch)/len(texts), 1.0))
        prog.empty()
        emb_array = np.array(all_emb)
        try:
            pickle.dump(emb_array, open(cache_file,'wb'))
        except:
            pass
        return emb_array

    def find_matches(self, external_courses, df, embeddings):
        results = {}
        st.info('Searching for similar courses...')
        for idx, course in enumerate(external_courses):
            title = course['title']
            desc = course['description']
            kw = course['keywords']
            lvl = course['target_level']
            df_f, emb_f = df, embeddings
            if kw:
                mask = df.apply(lambda r: any(k.strip().lower() in f"{r['course_code']} {r['course_title']} {r['course_description']}".lower() for k in kw.split(',')), axis=1)
                df_f = df[mask]
                emb_f = embeddings[mask.values]
                if df_f.empty:
                    st.warning(f'No matches for keywords: {kw}')
                    continue
            ext_emb = st.session_state.model.encode([f"{title} {desc}"])
            sims = cosine_similarity(ext_emb, emb_f)[0]
            if lvl:
                sims += df_f['course_level'].apply(lambda x: self.calculate_level_bonus(x, lvl)).values
            # take top 5
            idxs = np.argpartition(sims, -5)[-5:]
            top = idxs[np.argsort(sims[idxs])[::-1]]
            matches = []
            for i in top:
                row = df_f.iloc[i]
                matches.append({
                    'course_code': row['course_code'],
                    'title': row['course_title'],
                    'adjusted_similarity': sims[i]
                })
            if matches:
                results[idx] = matches
        return results

# Main app
def main():
    st.markdown('<h1 class="main-header">🎓 Will My Courses Transfer?</h1>', unsafe_allow_html=True)
    with st.sidebar:
        st.title('📋 Menu')
        if st.button('ℹ️ Show/Hide Help'):
            st.session_state.show_help = not st.session_state.show_help
        st.markdown('---')
        if st.session_state.model is None:
            if st.button('🚀 Start App'):
                with st.spinner('Loading model...'):
                    st.session_state.model = CourseTransferChecker().load_model()
        else:
            st.success('Model loaded')
        st.markdown('---')
        st.subheader('📁 Course Catalog')
