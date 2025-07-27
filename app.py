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
import os

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
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =======================
# HELPER FUNCTIONS
# =======================
def extract_level(code: str):
    """Extract course level from course code"""
    if not code: 
        return None
    try:
        m = re.search(r"(\d{3,4})", str(code))
        if not m: 
            return None
        n = int(m.group(1))
        return 100 if n < 200 else 200 if n < 300 else 300 if n < 400 else 400
    except (ValueError, AttributeError):
        return None

def level_bonus(orig, target):
    """Calculate level similarity bonus"""
    if orig is None or target is None: 
        return 0.0
    d = abs(orig - target)
    return 0.15 if d == 0 else 0.12 if d == 100 else 0.02 if d == 200 else 0.0

def calculate_transferability_score(t1, d1, t2, d2, model):
    """Calculate transferability score using logistic regression model"""
    try:
        # Encode descriptions and titles separately
        desc_embs = model.encode([d1, d2])
        sim_desc = cosine_similarity([desc_embs[0]], [desc_embs[1]])[0][0]
        
        title_embs = model.encode([t1, t2])
        sim_title = cosine_similarity([title_embs[0]], [title_embs[1]])[0][0]
        
        # Logistic regression coefficients (from training)
        # These should ideally be stored in a config file
        intercept = -7.144
        desc_coef = 9.219
        title_coef = 5.141
        
        # Calculate final probability using logistic function
        logit = intercept + (desc_coef * sim_desc) + (title_coef * sim_title)
        combined = 1 / (1 + math.exp(-logit))
        
        return sim_desc, sim_title, combined
    except Exception as e:
        st.error(f"Error calculating transferability score: {str(e)}")
        return 0, 0, 0

def get_category(score):
    """Categorize transferability score"""
    if score >= 0.85: 
        return "Very Likely", "🟢"
    elif score >= 0.73: 
        return "Likely", "🔵"
    elif score >= 0.6: 
        return "Needs Review", "🟡"
    else: 
        return "Unlikely", "🔴"

@st.cache_resource
def load_model():
    """Load and cache the sentence transformer model"""
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_csv(path):
    """Load and preprocess CSV file with proper encoding"""
    try:
        # Handle both file path string and uploaded file object
        if isinstance(path, str):
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found: {path}")
            df = pd.read_csv(path, encoding='latin1')
        else:
            # Uploaded file object
            df = pd.read_csv(path, encoding='latin1')
        
        # Check for required columns
        required_cols = ['course_title', 'course_description', 'course_code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean and preprocess
        df = df.dropna(subset=['course_title', 'course_description'])
        df['course_title'] = df['course_title'].astype(str).str.strip()
        df['course_description'] = df['course_description'].astype(str).str.strip()
        df['course_code'] = df['course_code'].astype(str).str.strip()
        df['level'] = df['course_code'].apply(extract_level)
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

@st.cache_data
def generate_embeddings(df, _model):
    """Generate embeddings for course catalog"""
    try:
        if _model is None:
            raise ValueError("Model is not loaded")
        
        texts = (df['course_code'] + ' ' + df['course_title'] + ' ' + df['course_description']).tolist()
        arr = _model.encode(texts, show_progress_bar=True)
        return np.array(arr)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def find_matches(external, model, df, embeddings):
    """Find course matches for external courses"""
    if model is None or df is None or embeddings is None:
        st.error("Missing required components for matching")
        return {}
    
    results = {}
    try:
        for idx, course in enumerate(external):
            title = course['title']
            desc = course['description']
            kw = course['keywords']
            lvl = course['target_level']
            
            # Generate embedding for external course
            ext_text = f"{title} {desc}"
            if kw:
                ext_text += f" {kw}"
            
            ext_emb = model.encode([ext_text])
            sims = cosine_similarity(ext_emb, embeddings)[0]
            
            # Add level bonus if specified
            if lvl:
                level_bonuses = df['level'].apply(lambda x: level_bonus(x, lvl)).values
                sims += level_bonuses
            
            # Get top 5 matches
            top_idx = np.argpartition(sims, -5)[-5:]
            sorted_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            
            matches = []
            for i in sorted_idx:
                row = df.iloc[i]
                sim_desc, sim_title, score = calculate_transferability_score(
                    title, desc, row['course_title'], row['course_description'], model)
                cat, emoji = get_category(score)
                
                matches.append({
                    'code': row['course_code'],
                    'title': row['course_title'],
                    'score': score,
                    'cat': cat,
                    'emoji': emoji,
                    'sim_desc': sim_desc,
                    'sim_title': sim_title,
                    'description': row['course_description'][:200] + "..." if len(row['course_description']) > 200 else row['course_description']
                })
            
            results[idx] = matches
    except Exception as e:
        st.error(f"Error finding matches: {str(e)}")
        return {}
    
    return results

# =======================
# UI LAYOUT
# =======================
def main():
    init_state()
    st.markdown('<h1 class="main-header">🎓 TransferzAI</h1>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.title("Controls")
        if st.button("Reset Application"):
            for k in ['model', 'courses_df', 'courses_emb', 'matches', 'external_courses']:
                if k in ['matches']:
                    st.session_state[k] = {}
                elif k in ['external_courses']:
                    st.session_state[k] = []
                else:
                    st.session_state[k] = None
            st.rerun()
        
        # Show memory usage info
        if st.session_state.courses_emb is not None:
            emb_size = st.session_state.courses_emb.nbytes / (1024 * 1024)  # MB
            st.info(f"Embeddings: {emb_size:.1f} MB")

    # Model loading
    if not st.session_state.model:
        st.write("### 🤖 Load AI Model")
        st.info("The AI model is required for semantic similarity analysis.")
        if st.button("Start AI Model"):
            with st.spinner("Loading sentence transformer model..."):
                model = load_model()
                if model is not None:
                    st.session_state.model = model
                    st.success("✅ Model ready!")
                    st.rerun()
    else:
        st.write("✅ **AI Model Loaded**")

    # Catalog loading
    st.markdown('<div class="step-header">📁 Load Course Catalog</div>', unsafe_allow_html=True)
    
    if st.session_state.model:
        src = st.radio("Data Source", ["W&M Catalog", "Upload CSV"])
        
        file_to_load = None
        if src == "W&M Catalog":
            default_file = "wm_courses_2025.csv"
            if os.path.exists(default_file):
                file_to_load = default_file
                st.info(f"Using default catalog: {default_file}")
            else:
                st.warning(f"Default catalog file '{default_file}' not found. Please upload a CSV file.")
                src = "Upload CSV"  # Force upload mode
        
        if src == "Upload CSV":
            uploaded_file = st.file_uploader("Upload Course Catalog CSV", type=['csv'])
            if uploaded_file:
                file_to_load = uploaded_file
        
        if file_to_load and st.button("Load Catalog"):
            with st.spinner("Loading and processing catalog..."):
                df = load_csv(file_to_load)
                if df is not None:
                    st.session_state.courses_df = df
                    
                    # Generate embeddings
                    with st.spinner("Generating course embeddings..."):
                        emb = generate_embeddings(df, st.session_state.model)
                        if emb is not None:
                            st.session_state.courses_emb = emb
                            st.success(f"✅ Loaded {len(df)} courses with embeddings generated.")
                        else:
                            st.error("Failed to generate embeddings.")
    else:
        st.info("Please load the AI model first.")

    # Show catalog overview
    if st.session_state.courses_df is not None:
        df = st.session_state.courses_df
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Courses", len(df))
        with col2:
            st.metric("Course Levels", df['level'].nunique())
        with col3:
            st.metric("Embeddings Ready", "Yes" if st.session_state.courses_emb is not None else "No")
        
        # Show sample courses
        with st.expander("Preview Catalog Data"):
            st.dataframe(df[['course_code', 'course_title', 'level']].head(10))

    # External courses input
    if st.session_state.courses_df is not None and st.session_state.courses_emb is not None:
        st.markdown('<div class="step-header">📚 Add External Courses</div>', unsafe_allow_html=True)
        st.info("Enter details for courses you want to find transfer equivalents for.")
        
        n = st.slider("Number of external courses", 1, 5, 2)
        external = []
        
        for i in range(n):
            with st.expander(f"External Course {i+1}", expanded=(i == 0)):
                col1, col2 = st.columns(2)
                with col1:
                    t = st.text_input("Course Title", key=f"t{i}", 
                                    placeholder="e.g., Introduction to Psychology")
                    k = st.text_input("Keywords (optional)", key=f"k{i}",
                                    placeholder="e.g., cognitive, behavior, research")
                with col2:
                    l = st.selectbox("Target Level", [None, 100, 200, 300, 400], key=f"l{i}",
                                   help="Course level for better matching")
                
                d = st.text_area("Course Description", key=f"d{i}", height=100,
                               placeholder="Detailed description of course content and objectives...")
                
                if t and d:
                    external.append({
                        'title': t, 
                        'description': d, 
                        'keywords': k, 
                        'target_level': l
                    })
        
        if external and st.button("🔍 Analyze Courses", type="primary"):
            with st.spinner("Finding course matches..."):
                matches = find_matches(external, st.session_state.model,
                                     st.session_state.courses_df,
                                     st.session_state.courses_emb)
                if matches:
                    st.session_state.external_courses = external
                    st.session_state.matches = matches
                    st.success("✅ Analysis complete!")
                    st.rerun()
    elif st.session_state.courses_df is not None:
        st.info("Please wait for embeddings to be generated before adding external courses.")

    # Show results
    if st.session_state.matches:
        st.markdown('<div class="step-header">🎯 Transfer Analysis Results</div>', unsafe_allow_html=True)
        
        # Summary statistics
        counts = {'Very Likely': 0, 'Likely': 0, 'Needs Review': 0, 'Unlikely': 0}
        
        for idx, matches in st.session_state.matches.items():
            ext_course = st.session_state.external_courses[idx]
            st.write(f"## External Course {idx+1}: {ext_course['title']}")
            
            if ext_course['target_level']:
                st.write(f"**Target Level:** {ext_course['target_level']}")
            
            st.write("**Top 5 Potential Matches:**")
            
            for rank, m in enumerate(matches, 1):
                pct = round(m['score'] * 100, 1)
                counts[m['cat']] += 1
                
                # Create expandable match card
                with st.expander(f"#{rank} {m['emoji']} **{m['cat']}** – {pct}% confidence → {m['code']}: {m['title']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Description Similarity", f"{m['sim_desc']:.3f}")
                        st.metric("Title Similarity", f"{m['sim_title']:.3f}")
                    with col2:
                        st.metric("Final Score", f"{m['score']:.3f}")
                        st.metric("Transferability", m['cat'])
                    
                    st.write("**Course Description:**")
                    st.write(m['description'])
            
            st.markdown("---")

        # Summary dashboard
        st.write("## 📊 Analysis Summary")
        total = sum(counts.values())
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Matches", total)
        with col2:
            st.metric("🟢 Very Likely", counts['Very Likely'])
        with col3:
            st.metric("🔵 Likely", counts['Likely'])
        with col4:
            st.metric("🟡 Needs Review", counts['Needs Review'])
        with col5:
            st.metric("🔴 Unlikely", counts['Unlikely'])
        
        # Export results button
        if st.button("📁 Export Results"):
            # Create results DataFrame for export
            export_data = []
            for idx, matches in st.session_state.matches.items():
                ext_course = st.session_state.external_courses[idx]
                for rank, m in enumerate(matches, 1):
                    export_data.append({
                        'External_Course': ext_course['title'],
                        'External_Description': ext_course['description'],
                        'Rank': rank,
                        'Match_Code': m['code'],
                        'Match_Title': m['title'],
                        'Transferability': m['cat'],
                        'Confidence_Score': round(m['score'] * 100, 1),
                        'Title_Similarity': round(m['sim_title'], 3),
                        'Description_Similarity': round(m['sim_desc'], 3)
                    })
            
            results_df = pd.DataFrame(export_data)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="transfer_analysis_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
