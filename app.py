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

# Custom CSS
st.markdown('''<style>
.main-header {font-size:2.5rem;color:#1f77b4;text-align:center;margin-bottom:2rem;}
.step-header {font-size:1.4rem;color:#2e8b57;margin:2rem 0;padding:10px;background:#f0f8f0;border-radius:8px;}
.help-text {background:#f8f9fa;padding:15px;border-radius:8px;border-left:4px solid #17a2b8;margin:10px 0;color:#333;}
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

    def extract_level(self, code:str):
        m = re.search(r'(\d{3,4})', code or '')
        if not m: return None
        n = int(m.group(1))
        return 100 if n<200 else 200 if n<300 else 300 if n<400 else 400

    def level_bonus(self, lvl_orig, lvl_target):
        if lvl_orig is None or lvl_target is None: return 0.0
        d=abs(lvl_orig-lvl_target)
        return 0.15 if d==0 else 0.12 if d==100 else 0.02 if d==200 else 0.0

    @st.cache_data
    def load_csv_data(_self, source_path:str):
        encs=['utf-8','latin-1','cp1252','iso-8859-1']
        df=None
        for enc in encs:
            try:
                path = 'wm_courses_2025.csv' if source_path=='url' else source_path
                df=pd.read_csv(path,encoding=enc)
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
        df=df.dropna(subset=['course_title','course_description'])
        df['level']=df['course_code'].apply(_self.extract_level)
        return df

    @st.cache_data
    def generate_embeddings(_self, df:pd.DataFrame):
        texts=(df['course_code']+' '+df['course_title']+' '+df['course_description']).tolist()
        # simple cache key
        import hashlib
        key=hashlib.md5('|'.join(texts).encode()).hexdigest()
        cache_file=Path(f'emb_{key}.pkl')
        if cache_file.exists():
            try: return pickle.load(open(cache_file,'rb'))
            except: pass
        model=st.session_state.model
        if not model:
            st.error('Model not loaded.')
            return None
        prog=st.progress(0)
        embs=[]
        batch=16
        for i in range(0,len(texts),batch):
            chunk=texts[i:i+batch]
            emb=model.encode(chunk,show_progress_bar=False)
            embs.extend(emb)
            prog.progress(min((i+batch)/len(texts),1.0))
        prog.empty()
        arr=np.array(embs)
        try: pickle.dump(arr, open(cache_file,'wb'))
        except: pass
        return arr

    def find_matches(self, external:list, df:pd.DataFrame, embs:np.ndarray):
        results={}
        st.info('🔍 Finding matches...')
        for idx,course in enumerate(external):
            title,desc,kw,lvl=course.values()
            sub_df,sub_emb=df,embs
            if kw:
                mask=df.apply(lambda r: any(k.strip().lower() in (r['course_code']+' '+r['course_title']+' '+r['course_description']).lower() for k in kw.split(',')),axis=1)
                sub_df,sub_emb=df[mask],embs[mask.values]
                if sub_df.empty:
                    st.warning(f'No courses for keywords: {kw}')
                    continue
            ext_emb=st.session_state.model.encode([f"{title} {desc}"])
            sims=cosine_similarity(ext_emb,sub_emb)[0]
            if lvl:
                sims+=sub_df['level'].apply(lambda x:self.level_bonus(x,lvl)).values
            # top 5
            top_idx=np.argpartition(sims,-5)[-5:]
            top=top_idx[np.argsort(sims[top_idx])[::-1]]
            hits=[{'code':sub_df.iloc[i]['course_code'],'title':sub_df.iloc[i]['course_title'],'sim':sims[i]} for i in top]
            if hits: results[idx]=hits
        return results

# Build UI

def main():
    st.markdown('<h1 class="main-header">🎓 Will My Courses Transfer?</h1>',unsafe_allow_html=True)
    checker=CourseTransferChecker()

    with st.sidebar:
        st.title('📋 Menu')
        if st.button('ℹ️ Show/Hide Help'):
            st.session_state.show_help=not st.session_state.show_help
        st.markdown('---')
        if not st.session_state.model:
            if st.button('🚀 Start Model'):
                with st.spinner('Loading...'):
                    st.session_state.model=checker.load_model()
        else:
            st.success('Model loaded')
        st.markdown('---')
        st.subheader('📁 Course Catalog')
        src=st.radio('Source',['Upload CSV','Built-in catalog'],key='src')
        if st.button('🔄 Reset All'):
            for k in ('model','courses_df','courses_emb','matches'): st.session_state[k]=None if k!='matches' else {}
            st.experimental_rerun()

    if st.session_state.show_help:
        st.markdown('<div class="help-text"><h3>How to Use</h3><ol><li>Start model</li><li>Load catalog</li><li>Add courses</li><li>Analyze transfer</li></ol></div>',unsafe_allow_html=True)

    # Load catalog
    st.markdown('<div class="step-header">📁 Step 1: Load Catalog</div>',unsafe_allow_html=True)
    if src=='Upload CSV':
        file=st.file_uploader('CSV',type='csv')
    else:
        file='url'; st.info('Using built-in')
    if file and st.session_state.model and st.button('📂 Load Catalog'):
        df=checker.load_csv_data('url' if src!='Upload CSV' else file)
        if df is not None:
            st.session_state.courses_df=df
            with st.spinner('Embedding...'):
                st.session_state.courses_emb=checker.generate_embeddings(df)
            st.success('Catalog ready')

    # Enter courses
    if st.session_state.courses_df is not None:
        st.markdown('<div class="step-header">📚 Step 2: Add Your Courses</div>',unsafe_allow_html=True)
        n=st.slider('How many?',1,10,3)
        external=[]
        for i in range(n):
            with st.expander(f'Course {i+1}',expanded=i<2):
                t=st.text_input('Title',key=f't{i}')
                d=st.text_area('Description',key=f'd{i}')
                k=st.text_input('Keywords',key=f'k{i}')
                l=st.selectbox('Level',[None,100,200,300,400],key=f'l{i}',format_func=lambda x:'Any' if x is None else f'{x}')
            if t and d: external.append({'title':t,'description':d,'keywords':k,'target_level':l})

        # Analyze
        if external and st.button('🔍 Analyze Courses'):
            st.session_state.matches=checker.find_matches(external,st.session_state.courses_df,st.session_state.courses_emb)

    # Display results
    if st.session_state.matches:
        st.markdown('<div class="step-header">✅ Results</div>',unsafe_allow_html=True)
        for idx,hits in st.session_state.matches.items():
            course=external[idx]['title']
            st.write(f"**{course}**")
            best=hits[0]
            score=best['sim']
            pct=round(score*100,1)
            if score>=TRANSFER_THRESHOLD:
                st.success(f"✅ Likely to Transfer ({pct}%) — {best['code']}: {best['title']}")
            else:
                st.warning(f"⚠️ Needs Review ({pct}%)")
                st.markdown('Top alternatives:')
                for alt in hits[:3]: st.markdown(f"- {alt['code']}: {alt['title']} ({round(alt['sim']*100,1)}%)")
            st.markdown('---')

if __name__=='__main__':
    main()
