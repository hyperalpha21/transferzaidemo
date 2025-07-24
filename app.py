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

# Page configuration
st.set_page_config(
    page_title="Course Transferability Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2e8b57;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.result-bar {
    height: 15px;
    background: #eee;
    border-radius: 10px;
    overflow: hidden;
    margin: 15px 0;
}
.result-bar-fill {
    height: 100%;
    transition: width 0.5s;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'university_courses_df' not in st.session_state:
    st.session_state.university_courses_df = None
if 'university_embeddings' not in st.session_state:
    st.session_state.university_embeddings = None
if 'all_comparisons' not in st.session_state:
    st.session_state.all_comparisons = []
if 'course_matches' not in st.session_state:
    st.session_state.course_matches = {}
if 'instructions_visible' not in st.session_state:
    st.session_state.instructions_visible = True

class CourseTransferabilityAnalyzer:
    def __init__(self):
        self.csv_url = "wm_courses_2025.csv"

    @st.cache_resource
    def load_model(_self):
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    def extract_course_level(self, course_code):
        try:
            match = re.search(r'(\d{3,4})', course_code)
            if match:
                number = int(match.group(1))
                if number < 200:
                    return 100
                elif number < 300:
                    return 200
                elif number < 400:
                    return 300
                else:
                    return 400
            return None
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

    def load_csv_data(self, csv_source):
        try:
            if csv_source == "url":
                if Path("wm_courses_2025.csv").exists():
                    df = pd.read_csv("wm_courses_2025.csv")
                else:
                    st.error("CSV file not found")
                    return None
            else:
                df = pd.read_csv(csv_source)

            required = ['course_code', 'course_title', 'course_description']
            if any(col not in df.columns for col in required):
                st.error("Missing required columns")
                return None

            df.dropna(subset=['course_title', 'course_description'], inplace=True)
            df['course_level'] = df['course_code'].apply(self.extract_course_level)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None

    def generate_embeddings(self, df, model):
        try:
            texts = [f"{row['course_code']} {row['course_title']} {row['course_description']}" for _, row in df.iterrows()]
            path = Path("wm_embeddings_cached.pkl")
            if path.exists():
                with open(path, 'rb') as f:
                    embs = pickle.load(f)
                if len(embs) == len(texts):
                    return embs
            embs = model.encode(texts, batch_size=32, show_progress_bar=False)
            with open(path, 'wb') as f:
                pickle.dump(embs, f)
            return embs
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return None

    def find_matches(self, external_courses, model, df, embeddings):
        matches = {}
        for i, course in enumerate(external_courses):
            title, desc, keywords, target = course['title'], course['description'], course['keywords'], course['target_level']
            if not title or not desc:
                continue
            # keyword filter
            if keywords.strip():
                kws = [k.strip().lower() for k in keywords.split(',')]
                df = df[df.apply(lambda r: any(k in f"{r['course_code']} {r['course_title']} {r['course_description']}`".lower() for k in kws), axis=1)]
            if len(df) == 0:
                continue
            texts = [f"{r['course_code']} {r['course_title']} {r['course_description']}" for _, r in df.iterrows()]
            embs = model.encode(texts) if len(df) < len(st.session_state.university_courses_df) else embeddings
            ext_emb = model.encode([f"{title} {desc}"])
            sims = cosine_similarity(ext_emb, embs)[0]
            if target:
                for idx in range(len(sims)):
                    sims[idx] += self.calculate_level_bonus(df.iloc[idx]['course_level'], target)
            top_idx = np.argsort(sims)[-5:][::-1]
            course_matches = []
            for idx in top_idx:
                row = df.iloc[idx]
                orig = cosine_similarity(ext_emb, [embs[idx]])[0][0]
                adj = sims[idx]
                bonus = adj - orig
                course_matches.append({
                    'course_code': row['course_code'],
                    'title': row['course_title'],
                    'description': row['course_description'],
                    'course_level': row['course_level'],
                    'similarity': orig,
                    'adjusted_similarity': adj,
                    'level_bonus': bonus
                })
            matches[i] = course_matches
        return matches

    def calculate_transferability(self, title1, desc1, title2, desc2, model):
        try:
            e_desc = model.encode([desc1, desc2])
            sim_desc = cosine_similarity([e_desc[0]], [e_desc[1]])[0][0]
            e_title = model.encode([title1, title2])
            sim_title = cosine_similarity([e_title[0]], [e_title[1]])[0][0]
            score = 1 / (1 + math.exp(-(-7.144 + 9.219 * sim_desc + 5.141 * sim_title)))
            return sim_desc, sim_title, score
        except:
            return None, None, None

    def get_transferability_category(self, score):
        if score >= 0.85: return "Very High Transferability", "🟢"
        elif score >= 0.7279793: return "Likely Transferable", "🔵"
        elif score >= 0.6: return "Possibly Transferable", "🟡"
        elif score >= 0.4: return "Unlikely Transferable", "🟠"
        return "Very Low Transferability", "🔴"

def main():
    st.markdown('<h1 class="main-header">🎓 Course Transferability Analyzer</h1>', unsafe_allow_html=True)
    analyzer = CourseTransferabilityAnalyzer()

    with st.sidebar:
        st.title("📋 Controls")
        if st.button("Toggle Instructions"):
            st.session_state.instructions_visible = not st.session_state.instructions_visible
        if st.session_state.model is None:
            if st.button("🔄 Load AI Model"):
                with st.spinner("Loading model..."):
                    st.session_state.model = analyzer.load_model()
                if st.session_state.model: st.success("✅ Model loaded!")
        st.subheader("📁 Data Source")
        csv_source = st.radio("Choose CSV source:", ["Upload File", "Use App's CSV File"], key="csv_source")
        if st.button("🔄 Clear Session"):
            for k in list(st.session_state.keys()):
                if k != 'instructions_visible': del st.session_state[k]
            st.experimental_rerun()

    if st.session_state.instructions_visible:
        st.info("1. Load AI model → 2. Load courses → 3. Add external courses → 4. Find & analyze matches")

    if st.session_state.model is None:
        st.warning("⚠️ Load the AI model first!")
        return

    st.subheader("📁 Step 1: Load Course Catalog")
    csv_file = st.file_uploader("Upload CSV", type=['csv']) if csv_source == "Upload File" else "url"
    if csv_file and st.button("Load Course Data"):
        with st.spinner("Loading courses..."):
            df = analyzer.load_csv_data(csv_file)
        if df is not None:
            st.session_state.university_courses_df = df
            with st.spinner("Generating embeddings..."):
                embs = analyzer.generate_embeddings(df, st.session_state.model)
            if embs is not None: st.session_state.university_embeddings = embs
            st.success("✅ Data ready!")
            st.dataframe(df[['course_code','course_title','course_level']].head(5))

    if st.session_state.university_courses_df is not None:
        st.subheader("📚 Step 2: Add External Courses")
        n = st.slider("Number of external courses",1,10,3)
        external_courses = []
        for i in range(n):
            with st.expander(f"External Course {i+1}"):
                title = st.text_input(f"Title {i+1}")
                desc = st.text_area(f"Description {i+1}")
                kw = st.text_input(f"Keywords {i+1} (optional)")
                lvl = st.selectbox(f"Target Level {i+1}",[None,100,200,300,400],format_func=lambda x:"Any" if x is None else f"{x} Level")
                if title and desc:
                    external_courses.append({'title':title,'description':desc,'keywords':kw,'target_level':lvl})

        if external_courses and st.button("Find Top 5 Matches"):
            with st.spinner("Finding matches..."):
                st.session_state.course_matches = analyzer.find_matches(external_courses,st.session_state.model,st.session_state.university_courses_df,st.session_state.university_embeddings)

    if st.session_state.course_matches:
        st.subheader("🎯 Step 4: Analyze Transferability")
        sel = {}
        for i, matches in st.session_state.course_matches.items():
            st.write(f"Select matches for External Course {i+1}")
            opts = [f"{m['course_code']} - {m['title']}" for m in matches]
            chosen = st.multiselect(f"Choose for analysis {i+1}", opts)
            if chosen: sel[i] = [opts.index(c) for c in chosen]

        if sel and st.button("Analyze Selected Matches"):
            results = []
            for ci, mis in sel.items():
                ext = external_courses[ci]
                for mi in mis:
                    m = st.session_state.course_matches[ci][mi]
                    ds,ts,score = analyzer.calculate_transferability(ext['title'],ext['description'],m['title'],m['description'],st.session_state.model)
                    if score is not None:
                        cat,emoji = analyzer.get_transferability_category(score)
                        results.append({
                            'External': ext['title'],
                            'WM Code': m['course_code'],
                            'WM Title': m['title'],
                            'Score': score,
                            'Category': cat,
                            'Emoji': emoji
                        })
            if results:
                st.success("✅ Analysis complete!")
                # Show big cards
                color_map = {
                    "Very High Transferability":"#d4edda",
                    "Likely Transferable":"#cce5ff",
                    "Possibly Transferable":"#fff3cd",
                    "Unlikely Transferable":"#ffe5b4",
                    "Very Low Transferability":"#f8d7da"
                }
                for r in results:
                    bg=color_map[r['Category']]
                    bar_color='#28a745' if r['Score']>0.7 else ('#ffc107' if r['Score']>0.4 else '#dc3545')
                    st.markdown(f"""
                    <div class='result-card' style='background:{bg};'>
                        <h2 style='text-align:center;'>{r['Emoji']} {r['Category']}</h2>
                        <p style='text-align:center;'><b>{r['External']}</b> → <b>{r['WM Code']} {r['WM Title']}</b></p>
                        <div class='result-bar'><div class='result-bar-fill' style='width:{r['Score']*100}%;background:{bar_color};'></div></div>
                        <p style='text-align:center;'>Transferability Score: <b>{round(r['Score']*100,1)}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                # Collapsible table
                with st.expander("📊 View Full Table"):
                    st.dataframe(pd.DataFrame(results))

if __name__ == "__main__":
    main()
