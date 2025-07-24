import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
from datetime import datetime
import pickle
from pathlib import Path

# Threshold for auto-classifying transferability\ nTRANSFER_THRESHOLD = 0.75

# Page configuration
st.set_page_config(
    page_title='Welcome to TransferzAI',
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS - simplified and cleaner
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
.help-text h3 {
    color: #2c3e50;
    margin-bottom: 10px;
}
.help-text ol {
    color: #333333;
}
.help-text li {
    color: #333333;
    margin-bottom: 5px;
}
.help-text strong {
    color: #2c3e50;
}
</style>''', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'university_courses_df' not in st.session_state:
    st.session_state.university_courses_df = None
if 'university_embeddings' not in st.session_state:
    st.session_state.university_embeddings = None
if 'course_matches' not in st.session_state:
    st.session_state.course_matches = {}
if 'show_help' not in st.session_state:
    st.session_state.show_help = True

class CourseTransferChecker:
    def __init__(self):
        self.csv_url = 'wm_courses_2025.csv'

    @st.cache_resource
    def load_model(_self):
        try:
            return SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            st.error(f'Could not load model: {e}')
            return None

    def extract_course_level(self, course_code):
        try:
            match = re.search(r'(\d{3,4})', course_code)
            if not match:
                return None
            number = int(match.group(1))
            if number < 200:
                return 100
            elif number < 300:
                return 200
            elif number < 400:
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
    def load_csv_data(_self, csv_source):
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for enc in encodings:
                try:
                    if csv_source == 'url':
                        if Path('wm_courses_2025.csv').exists():
                            df = pd.read_csv('wm_courses_2025.csv', encoding=enc)
                        else:
                            st.error('Catalog file not found.')
                            return None
                    else:
                        df = pd.read_csv(csv_source, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error('Could not read CSV; unsupported format.')
                return None

            required = ['course_code', 'course_title', 'course_description']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f'Missing columns: {missing}')
                return None

            df.dropna(subset=['course_title', 'course_description'], inplace=True)
            df['course_level'] = df['course_code'].apply(_self.extract_course_level)
            return df
        except Exception as e:
            st.error(f'Error loading catalog: {e}')
            return None

    @st.cache_data
    def generate_embeddings(_self, df, model):
        try:
            texts = [f"{row['course_code']} {row['course_title']} {row['course_description']}" for _, row in df.iterrows()]
            import hashlib
            h = hashlib.md5(str(sorted(texts)).encode()).hexdigest()
            cache_file = Path(f'embeddings_cache_{h}.pkl')
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if len(data) == len(texts):
                    return data
            st.info('Processing course descriptions...')
            prog = st.progress(0)
            batch = 16
            total = (len(texts) + batch - 1) // batch
            all_emb = []
            for i in range(0, len(texts), batch):
                chunk = texts[i:i + batch]
                emb = model.encode(chunk, batch_size=batch, show_progress_bar=False)
                all_emb.extend(emb)
                prog.progress((i // batch + 1) / total)
            prog.empty()
            arr = np.array(all_emb)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(arr, f)
            except:
                pass
            return arr
        except Exception as e:
            st.error(f'Error generating embeddings: {e}')
            return None

    def find_matches(self, external_courses, model, df, embeddings):
        matches = {}
        st.info('🔍 Searching for similar courses...')
        for idx, course in enumerate(external_courses):
            title = course['title']; desc = course['description']
            kw = course['keywords']; lvl = course['target_level']
            if not title or not desc:
                continue

            filt_df = df; filt_emb = embeddings
            if kw.strip():
                keys = [k.strip().lower() for k in kw.split(',')]
                mask = df.apply(lambda r: any(k in f"{r['course_code']} {r['course_title']} {r['course_description']}".lower() for k in keys), axis=1)
                filt_df = df[mask]; filt_emb = embeddings[mask.values]
                if filt_df.empty:
                    st.warning(f'No courses matching keywords: {kw}')
                    continue

            ext_text = f"{title} {desc}"
            ext_emb = model.encode([ext_text])
            sims = cosine_similarity(ext_emb, filt_emb)[0]
            if lvl:
                bonus = filt_df['course_level'].apply(lambda x: self.calculate_level_bonus(x, lvl)).values
                sims += bonus

            top_idx = np.argpartition(sims, -5)[-5:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            lst = []
            for i in top_idx:
                row = filt_df.iloc[i]
                adj = sims[i]
                lst.append({'course_code': row['course_code'], 'title': row['course_title'], 'description': row['course_description'], 'course_level': row['course_level'], 'adjusted_similarity': adj})
            matches[idx] = lst
        return matches

def main():
    st.markdown('<h1 class=\'main-header\'>🎓 Will My Courses Transfer?</h1>', unsafe_allow_html=True)
    with st.sidebar:
        st.title('📋 Menu')
        if st.button('ℹ️ Show/Hide Help'):
            st.session_state.show_help = not st.session_state.show_help
        st.markdown('---')
        if st.session_state.model is None:
            if st.button('🚀 Start the App'):
                with st.spinner('Loading model...'):
                    analyzer = CourseTransferChecker()
                    st.session_state.model = analyzer.load_model()
                if st.session_state.model:
                    st.success('✅ Ready to go!')
                else:
                    st.error('❌ Load error')
        else:
            st.success('✅ App is ready!')
        st.markdown('---')
        st.subheader('📁 Course Catalog')
        csv_src = st.radio('Where are your university courses?', ['Upload a file', 'Use W&M catalog'], key='csv_source')
        st.markdown('---')
        if st.button('🔄 Start Over'):
            for k in list(st.session_state.keys()):
                if k != 'show_help':
                    del st.session_state[k]
            st.experimental_rerun()

    if st.session_state.show_help:
        st.markdown('<div class=\'help-text\'><h3>How to Use This App:</h3><ol><li><strong>Start the App:</strong> Click "Start the App"</li><li><strong>Load Courses:</strong> Upload or use catalog</li><li><strong>Add Your Courses:</strong> Enter courses to transfer</li><li><strong>Analyze:</strong> Click “Analyze Courses”</li></ol></div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning('⚠️ Please start the app first!')
        return

    analyzer = CourseTransferChecker()
    # Step 1: Load catalog
    st.markdown('<div class=\'step-header\'>📁 Step 1: Load University Course Catalog</div>', unsafe_allow_html=True)
    if csv_src == 'Upload a file':
        st.info('Upload a CSV with columns: course_code, course_title, course_description')
        csv_file = st.file_uploader('Choose CSV', type=['csv'])
    else:
        csv_file = 'url'; st.info('Using W&M catalog')

    if csv_file and st.button('📂 Load Courses'):
        with st.spinner('Loading catalog...'):
            df = analyzer.load_csv_data(csv_file)
        if df is not None:
            st.session_state.university_courses_df = df
            with st.spinner('Preparing embeddings...'):
                embs = analyzer.generate_embeddings(df, st.session_state.model)
            if embs is not None:
                st.session_state.university_embeddings = embs
                st.success('✅ Course catalog ready!')
                with st.expander('📋 Preview'):
                    st.dataframe(df[['course_code','course_title','course_level']].head(10), use_container_width=True)

    # Step 2: Enter external courses
    if st.session_state.university_courses_df is not None:
        st.markdown('<div class=\'step-header\'>📚 Step 2: Enter Your Courses</div>', unsafe_allow_html=True)
        num = st.slider('How many courses to check?',1,10,3)
        external_courses = []
        for i in range(num):
            with st.expander(f'Course #{i+1}', expanded=i<2):
                c1,c2 = st.columns([2,1])
                with c1:
                    t = st.text_input('Course Title', key=f'title_{i}',placeholder='e.g., Intro to Psychology')
                    d = st.text_area('Course Description', key=f'desc_{i}',height=100)
                with c2:
                    kw = st.text_input('Keywords (optional)', key=f'kw_{i}',placeholder='psychology, behavior')
                    lvl = st.selectbox('Course Level',[None,100,200,300,400], format_func=lambda x: 'Any' if x is None else f'{x} Level',key=f'level_{i}')
                if t and d:
                    external_courses.append({'title':t,'description':d,'keywords':kw,'target_level':lvl})

        # Step 3: Analyze
        if external_courses:
            st.markdown('<div class=\'step-header\'>🔍 Step 3: Check Transferability</div>', unsafe_allow_html=True)
            if st.button('🔍 Analyze Courses', type='primary'):
                with st.spinner('Analyzing...'):
                    m = analyzer.find_matches(external_courses, st.session_state.model, st.session_state.university_courses_df, st.session_state.university_embeddings)
                    st.session_state.course_matches = m
                if m: st.success('✅ Analysis complete!')

    # Step 4: Display results
    if st.session_state.course_matches:
        st.markdown('<div class=\'step-header\'>✅ Results: Transferability</div>', unsafe_allow_html=True)
        for idx, lst in st.session_state.course_matches.items():
            ext = external_courses[idx]
            st.write(f"**Your Course:** {ext['title']}")
            top = lst[0]
            score = top['adjusted_similarity']
            pct = round(score*100,1)
            if score >= TRANSFER_THRESHOLD:
                st.success(f"✅ Likely to Transfer ({pct}% match)")
                st.markdown(f"**Best match:** {top['course_code']} - {top['title']}")
            else:
                st.warning(f"⚠️ Needs Advisor Review ({pct}% match)")
                st.markdown('Here are the top 3 possible matches for transparency:')
                for alt in lst[:3]:
                    alt_pct = round(alt['adjusted_similarity']*100,1)
                    st.markdown(f"- **{alt['course_code']} - {alt['title']}** ({alt_pct}% match)")
            st.markdown('---')

if __name__=='__main__':
    main()
