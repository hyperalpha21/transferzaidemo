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
.help-text h3 { color: #2c3e50; margin-bottom: 10px; }
.help-text ol, .help-text li { color: #333333; }
.help-text strong { color: #2c3e50; }
</style>''', unsafe_allow_html=True)

# Session state initialization
for key in ('model','university_courses_df','university_embeddings','course_matches','show_help'):
    if key not in st.session_state:
        st.session_state[key] = {} if key == 'course_matches' else (True if key=='show_help' else None)

class CourseTransferChecker:
    @st.cache_resource
    def load_model(self):
        try:
            return SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            st.error(f'Error loading model: {e}')
            return None

    def extract_course_level(self, code):
        try:
            num = re.search(r'(\d{3,4})', code)
            if not num: return None
            level = int(num.group(1))
            return (100 if level<200 else 200 if level<300 else 300 if level<400 else 400)
        except:
            return None

    def calculate_level_bonus(self, course_level, target_level):
        if course_level is None or target_level is None:
            return 0.0
        diff = abs(course_level - target_level)
        return 0.15 if diff==0 else 0.12 if diff==100 else 0.02 if diff==200 else 0.0

    @st.cache_data
    def load_csv_data(self, source):
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
                st.error('Could not load CSV.'); return None
            required = ['course_code','course_title','course_description']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f'Missing columns: {missing}'); return None
            df.dropna(subset=['course_title','course_description'], inplace=True)
            df['course_level'] = df['course_code'].apply(self.extract_course_level)
            return df
        except Exception as e:
            st.error(f'Error loading catalog: {e}')
            return None

    @st.cache_data
    def generate_embeddings(self, df):
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
        batch = 16
        all_emb = []
        for i in range(0,len(texts),batch):
            chunk = texts[i:i+batch].tolist()
            emb = model.encode(chunk, show_progress_bar=False)
            all_emb.extend(emb)
            prog.progress(min((i+batch)/len(texts),1.0))
        emb_array = np.array(all_emb)
        try: pickle.dump(emb_array, open(cache_file,'wb'))
        except: pass
        prog.empty()
        return emb_array

    def find_matches(self, external_courses, df, embeddings):
        results = {}
        st.info('Searching for similar courses...')
        for idx, course in enumerate(external_courses):
            title, desc, kw, lvl = course.values()
            df_f, emb_f = df, embeddings
            if kw:
                mask = df.apply(lambda r: any(k.lower() in (r['course_code']+r['course_title']+r['course_description']).lower() for k in kw.split(',')), axis=1)
                df_f, emb_f = df[mask], embeddings[mask.values]
                if df_f.empty:
                    st.warning(f'No keyword matches for {kw}'); continue
            emb_ext = st.session_state.model.encode([f"{title} {desc}"])
            sims = cosine_similarity(emb_ext, emb_f)[0]
            if lvl:
                sims += df_f['course_level'].apply(lambda x: self.calculate_level_bonus(x,lvl)).values
            idxs = np.argpartition(sims,-5)[-5:]
            top = idxs[np.argsort(sims[idxs])[::-1]]
            matches = [{'course_code': df_f.iloc[i]['course_code'], 'title':df_f.iloc[i]['course_title'], 'adjusted_similarity':sims[i]} for i in top]
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
        source = st.radio('Catalog source',['Upload file','W&M catalog'], key='csv_source')
        if st.button('🔄 Reset'):
            for k in list(st.session_state.keys()):
                if k not in ['show_help']:
                    del st.session_state[k]
            st.experimental_rerun()

    if st.session_state.show_help:
        st.markdown('<div class="help-text"><h3>How to use:</h3><ol><li>Start App</li><li>Load catalog</li><li>Enter courses</li><li>Analyze</li></ol></div>', unsafe_allow_html=True)
    if st.session_state.model is None:
        st.warning('Start the app first'); return
    checker = CourseTransferChecker()

    # Step 1: Load catalog
    st.markdown('<div class="step-header">📁 Step 1: Load Catalog</div>', unsafe_allow_html=True)
    if source=='Upload file':
        file = st.file_uploader('CSV file', type='csv')
    else:
        file = 'url'; st.info('Using built-in catalog')
    if file and st.button('📂 Load'):
        df = checker.load_csv_data('url' if source!=' ' else file)
        if df is not None:
            st.session_state.university_courses_df = df
            with st.spinner('Embedding...'):
                emb = checker.generate_embeddings(df)
            st.session_state.university_embeddings = emb
            st.success('Catalog ready')
            with st.expander('Preview'):
                st.dataframe(df[['course_code','course_title']].head(5))

    # Step 2: Enter courses
    if st.session_state.university_courses_df is not None:
        st.markdown('<div class="step-header">📚 Step 2: Enter Your Courses</div>', unsafe_allow_html=True)
        n = st.slider('Number of courses',1,10,3)
        external = []
        for i in range(n):
            with st.expander(f'Course {i+1}', expanded=i<2):
                t = st.text_input('Title', key=f't{i}')
                d = st.text_area('Description', key=f'd{i}')
                k = st.text_input('Keywords', key=f'k{i}')
                l = st.selectbox('Level',[None,100,200,300,400], key=f'l{i}')
            if t and d:
                external.append({'title':t,'description':d,'keywords':k,'target_level':l})

        # Step 3: Analyze
        if external:
            st.markdown('<div class="step-header">🔍 Step 3: Analyze</div>', unsafe_allow_html=True)
            if st.button('🔍 Analyze Courses'):
                res = checker.find_matches(external, st.session_state.university_courses_df, st.session_state.university_embeddings)
                st.session_state.course_matches = res
                st.success('Done')

    # Step 4: Display results
    if st.session_state.course_matches:
        st.markdown('<div class="step-header">✅ Results</div>', unsafe_allow_html=True)
        for idx, matches in st.session_state.course_matches.items():
            course = external[idx]['title']
            st.write(f"**{course}**")
            top = matches[0]
            score = top['adjusted_similarity']
            pct = round(score*100,1)
            if score>=TRANSFER_THRESHOLD:
                st.success(f"✅ Likely ({pct}%) – {top['course_code']}: {top['title']}")
            else:
                st.warning(f"⚠️ Review ({pct}%)")
                st.markdown('Top 3 alternatives:')
                for alt in matches[:3]:
                    ap = round(alt['adjusted_similarity']*100,1)
                    st.markdown(f"- {alt['course_code']}: {alt['title']} ({ap}%)")
            st.markdown('---')

if __name__=='__main__':
    main()
