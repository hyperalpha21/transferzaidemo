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
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'model': None,
        'courses_df': None,
        'courses_emb': None,
        'matches': {},
        'show_help': True,
        'external_courses': []
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# =======================
# CORE LOGIC FUNCTIONS
# =======================
def calculate_transferability_score(title_ext, desc_ext, title_match, desc_match, model):
    """Compute description + title similarity and combined logit transferability score"""
    try:
        # Ensure inputs are strings and not empty
        title_ext = str(title_ext) if title_ext else "No Title"
        desc_ext = str(desc_ext) if desc_ext else "No Description"
        title_match = str(title_match) if title_match else "No Title"
        desc_match = str(desc_match) if desc_match else "No Description"
        
        # encode descriptions
        desc_embs = model.encode([desc_ext, desc_match])
        sim_desc = cosine_similarity([desc_embs[0]], [desc_embs[1]])[0][0]

        # encode titles
        title_embs = model.encode([title_ext, title_match])
        sim_title = cosine_similarity([title_embs[0]], [title_embs[1]])[0][0]

        # multi-feature logistic regression (from your 2nd script)
        combined_score = 1/(1 + math.exp(-(-7.144 + 9.219 * sim_desc + 5.141 * sim_title)))
        return float(sim_desc), float(sim_title), float(combined_score)
    except Exception as e:
        st.error(f"Transferability calculation error: {e}")
        return 0.0, 0.0, 0.0

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
    """Extract course level from course code"""
    if not code:
        return None
    m = re.search(r"(\d{3,4})", str(code))
    if not m:
        return None
    n = int(m.group(1))
    if n < 200:
        return 100
    elif n < 300:
        return 200
    elif n < 400:
        return 300
    else:
        return 400

def level_bonus(orig, target):
    """Calculate level bonus based on course level difference"""
    if orig is None or target is None:
        return 0.0
    d = abs(orig - target)
    if d == 0:
        return 0.15
    elif d == 100:
        return 0.12
    elif d == 200:
        return 0.02
    else:
        return 0.0

# =======================
# DATA LOADING FUNCTIONS
# =======================
@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

@st.cache_data
def load_csv_data(source_path):
    """Load CSV data with multiple encoding attempts"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    # Handle different source types
    if source_path == 'url':
        # Create a sample dataset if no URL is available
        st.warning("URL source not available. Creating sample dataset.")
        df = create_sample_dataset()
    elif hasattr(source_path, 'read'):
        # Handle uploaded file
        for enc in encodings:
            try:
                source_path.seek(0)  # Reset file pointer
                df = pd.read_csv(source_path, encoding=enc)
                break
            except Exception:
                continue
    else:
        # Handle file path
        for enc in encodings:
            try:
                df = pd.read_csv(source_path, encoding=enc)
                break
            except Exception:
                continue
    
    if df is None:
        st.error('Could not read course catalog.')
        return None
    
    # Validate required columns
    required_columns = ['course_code', 'course_title', 'course_description']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f'Missing columns: {missing_columns}')
        return None
    
    # Clean data
    df = df.dropna(subset=['course_title', 'course_description'])
    df['course_code'] = df['course_code'].fillna('N/A')
    df['course_title'] = df['course_title'].astype(str)
    df['course_description'] = df['course_description'].astype(str)
    df['level'] = df['course_code'].apply(extract_level)
    
    return df

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    sample_data = {
        'course_code': ['MATH101', 'PHYS201', 'CHEM150', 'BIOL120', 'HIST200', 'ENG101', 'PSYC110', 'ECON201'],
        'course_title': [
            'Introduction to Calculus',
            'General Physics I',
            'General Chemistry',
            'Introduction to Biology',
            'World History',
            'English Composition',
            'Introduction to Psychology',
            'Microeconomics'
        ],
        'course_description': [
            'Introduction to differential and integral calculus with applications.',
            'Mechanics, heat, and sound. Laboratory included.',
            'Fundamental principles of chemistry including atomic structure and bonding.',
            'Basic principles of biology including cell structure and function.',
            'Survey of world civilizations from ancient times to present.',
            'Development of writing skills through analysis of texts.',
            'Introduction to psychological principles and research methods.',
            'Supply and demand, market structures, and consumer behavior.'
        ]
    }
    return pd.DataFrame(sample_data)

@st.cache_data
def generate_embeddings(df: pd.DataFrame, _model):
    """Generate embeddings for the course catalog"""
    if _model is None:
        st.error("Model not loaded")
        return None
        
    try:
        texts = (df['course_code'].astype(str) + ' ' + 
                df['course_title'].astype(str) + ' ' + 
                df['course_description'].astype(str)).tolist()
        
        # Create cache key
        import hashlib
        key = hashlib.md5('|'.join(texts).encode()).hexdigest()
        
        # Try to use temp directory for cache
        try:
            cache_dir = Path(tempfile.gettempdir()) / "transferzai_cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f'emb_{key}.pkl'
        except:
            cache_file = None
        
        # Try to load from cache
        if cache_file and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        # Generate embeddings
        prog = st.progress(0, text="Generating embeddings...")
        embs = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            emb = _model.encode(chunk, show_progress_bar=False)
            embs.extend(emb)
            progress = min((i + batch_size) / len(texts), 1.0)
            prog.progress(progress, text=f"Generating embeddings... {int(progress*100)}%")
        
        prog.empty()
        arr = np.array(embs)
        
        # Try to cache the result
        if cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(arr, f)
            except:
                pass
        
        return arr
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

# =======================
# FIND MATCHES WITH LOGIT
# =======================
def find_matches_with_logit(external_courses, model, df, embeddings):
    """Find top 5 matches by adjusted similarity, then compute logit-based transferability"""
    if model is None or df is None or embeddings is None:
        st.error("Missing required components for analysis")
        return {}
        
    results = {}

    for idx, course in enumerate(external_courses):
        try:
            title_ext = str(course.get('title', ''))
            desc_ext = str(course.get('description', ''))
            kw = course.get('keywords', '')
            target_level = course.get('target_level')

            if not title_ext.strip() or not desc_ext.strip():
                st.warning(f"Course {idx+1}: Missing title or description")
                continue

            # Start with full dataset
            sub_df = df.copy()
            sub_emb = embeddings.copy()
            
            # Apply keyword filtering if keywords provided
            if kw and kw.strip():
                keywords = [k.strip().lower() for k in kw.split(',') if k.strip()]
                if keywords:
                    mask = df.apply(
                        lambda r: any(
                            keyword in (str(r['course_code']) + ' ' + 
                                      str(r['course_title']) + ' ' + 
                                      str(r['course_description'])).lower()
                            for keyword in keywords
                        ), axis=1
                    )
                    
                    if mask.any():
                        sub_df = df[mask].copy()
                        sub_emb = embeddings[mask.values]
                    else:
                        st.warning(f"Course {idx+1}: No matches for keywords: {kw}")
                        continue

            if sub_df.empty:
                continue

            # Encode external course
            ext_text = f"{title_ext} {desc_ext}"
            ext_emb = model.encode([ext_text])
            
            # Calculate similarities
            sims = cosine_similarity(ext_emb, sub_emb)[0]

            # Apply level bonus
            if target_level:
                level_bonus_array = sub_df['level'].apply(
                    lambda x: level_bonus(x, target_level)
                ).to_numpy()
                sims = sims + level_bonus_array

            # Get top 5 matches
            num_matches = min(5, len(sims))
            top_idx = np.argpartition(sims, -num_matches)[-num_matches:]
            top_sorted = top_idx[np.argsort(sims[top_idx])[::-1]]

            matches = []
            for i in top_sorted:
                try:
                    row = sub_df.iloc[i]

                    # Calculate original similarity (without level bonus)
                    original_sim = cosine_similarity(ext_emb, [sub_emb[i]])[0][0]
                    adjusted_sim = sims[i]
                    lvl_bonus = adjusted_sim - original_sim

                    # Calculate transferability score
                    sim_desc, sim_title, combined_score = calculate_transferability_score(
                        title_ext, desc_ext,
                        row['course_title'], row['course_description'],
                        model
                    )
                    
                    category, emoji = get_transferability_category(combined_score)

                    matches.append({
                        'code': str(row['course_code']),
                        'title': str(row['course_title']),
                        'sim_original': float(original_sim),
                        'sim_adjusted': float(adjusted_sim),
                        'level_bonus': float(lvl_bonus),
                        'sim_desc': float(sim_desc),
                        'sim_title': float(sim_title),
                        'transfer_score': float(combined_score),
                        'category': category,
                        'emoji': emoji
                    })
                except Exception as e:
                    st.warning(f"Error processing match {i}: {e}")
                    continue
            
            # Sort by transfer score
            results[idx] = sorted(matches, key=lambda m: m['transfer_score'], reverse=True)
            
        except Exception as e:
            st.error(f"Error processing course {idx+1}: {e}")
            continue
    
    return results

# =======================
# STREAMLIT MAIN UI
# =======================
def main():
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🎓 Welcome to TransferzAI</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📋 Menu")
        
        if st.button("ℹ️ Show/Hide Help"):
            st.session_state.show_help = not st.session_state.show_help
        
        st.markdown("---")
        
        # Model loading
        if not st.session_state.model:
            if st.button("🚀 Start Model"):
                with st.spinner("Loading model..."):
                    st.session_state.model = load_model()
                    if st.session_state.model:
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to load model")
        else:
            st.success("✅ Model loaded")
        
        st.markdown("---")
        
        # Course catalog selection
        st.subheader("📁 Course Catalog")
        src = st.radio("Source", ["Upload CSV", "Sample Dataset"], key="src")
        
        if st.button("🔄 Reset All"):
            for key in ['model', 'courses_df', 'courses_emb', 'matches', 'external_courses']:
                if key == 'matches' or key == 'external_courses':
                    st.session_state[key] = {} if key == 'matches' else []
                else:
                    st.session_state[key] = None
            st.rerun()

    # Help instructions
    if st.session_state.show_help:
        st.markdown("""
        <div style='background:#f8f9fa;padding:1rem;border-left:4px solid #1f77b4'>
        <h3>📖 How to Use TransferzAI</h3>
        <ol>
            <li><strong>Start the Model</strong> in the sidebar (required for analysis)</li>
            <li><strong>Load Course Catalog</strong> (upload CSV or use sample dataset)</li>
            <li><strong>Add External Courses</strong> with title, description, and optional filters</li>
            <li><strong>Analyze</strong> to find matching courses with transferability scores</li>
        </ol>
        <p><strong>CSV Format:</strong> Must contain columns: course_code, course_title, course_description</p>
        </div>
        """, unsafe_allow_html=True)

    # Step 1: Load catalog
    st.markdown('<div class="step-header">📁 Step 1: Load Course Catalog</div>', unsafe_allow_html=True)
    
    file_to_process = None
    if src == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            file_to_process = uploaded_file
            st.info(f"📄 File uploaded: {uploaded_file.name}")
    else:
        file_to_process = "sample"
        st.info("📊 Using sample dataset for demonstration")
    
    if file_to_process and st.session_state.model and st.button("📂 Load Catalog"):
        with st.spinner("Loading catalog..."):
            df = load_csv_data(file_to_process)
            if df is not None:
                st.session_state.courses_df = df
                st.success(f"✅ Loaded {len(df)} courses")
                
                with st.spinner("Generating embeddings..."):
                    embeddings = generate_embeddings(df, st.session_state.model)
                    if embeddings is not None:
                        st.session_state.courses_emb = embeddings
                        st.success("✅ Embeddings generated successfully!")
                    else:
                        st.error("❌ Failed to generate embeddings")

    # Show catalog info if loaded
    if st.session_state.courses_df is not None:
        with st.expander("📊 Catalog Information"):
            df = st.session_state.courses_df
            st.write(f"**Total Courses:** {len(df)}")
            st.write(f"**Sample Courses:**")
            st.dataframe(df[['course_code', 'course_title']].head(), use_container_width=True)

    # Step 2: Add external courses
    external_courses = []
    if st.session_state.courses_df is not None and st.session_state.model is not None:
        st.markdown('<div class="step-header">📚 Step 2: Add Your External Courses</div>', unsafe_allow_html=True)
        
        num_courses = st.slider("Number of courses to analyze", 1, 10, 3)
        
        for i in range(num_courses):
            with st.expander(f"📝 Course {i+1}", expanded=(i < 2)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    title = st.text_input("Course Title", key=f"title_{i}", 
                                        placeholder="e.g., Introduction to Computer Science")
                    description = st.text_area("Course Description", key=f"desc_{i}", height=100,
                                             placeholder="Detailed description of the course content...")
                
                with col2:
                    keywords = st.text_input("Keywords (optional)", key=f"keywords_{i}",
                                           placeholder="math, science, programming")
                    level = st.selectbox("Target Level", [None, 100, 200, 300, 400], key=f"level_{i}",
                                       format_func=lambda x: "Any Level" if x is None else f"{x}-level")
                
                if title and description:
                    external_courses.append({
                        'title': title.strip(),
                        'description': description.strip(),
                        'keywords': keywords.strip() if keywords else '',
                        'target_level': level
                    })

        # Store external courses in session state
        st.session_state.external_courses = external_courses

        # Step 3: Analyze
        if external_courses:
            st.markdown('<div class="step-header">🔍 Step 3: Analyze Transferability</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Analyze Courses", type="primary"):
                with st.spinner("Analyzing course transferability..."):
                    matches = find_matches_with_logit(
                        external_courses,
                        st.session_state.model,
                        st.session_state.courses_df,
                        st.session_state.courses_emb
                    )
                    st.session_state.matches = matches
                    if matches:
                        st.success("✅ Analysis complete!")
                    else:
                        st.warning("⚠️ No matches found. Try adjusting your course descriptions or keywords.")

    # Step 4: Display Results
    if st.session_state.matches:
        st.markdown('<div class="step-header">📊 Step 4: Transfer Analysis Results</div>', unsafe_allow_html=True)
        
        for idx, matches in st.session_state.matches.items():
            if idx < len(st.session_state.external_courses):
                external_course = st.session_state.external_courses[idx]
                
                st.subheader(f"📘 External Course {idx+1}: {external_course['title']}")
                
                if not matches:
                    st.warning("No suitable matches found for this course.")
                    continue
                
                # Display matches
                for rank, match in enumerate(matches, 1):
                    pct = round(match['transfer_score'] * 100, 1)
                    
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **#{rank} {match['emoji']} {match['category']}** ({pct}% probability)
                            
                            **{match['code']}**: {match['title']}
                            """)
                        
                        with col2:
                            st.metric("Transfer Score", f"{pct}%")
                        
                        # Detailed metrics
                        with st.expander("📈 Detailed Metrics"):
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            
                            with metrics_col1:
                                st.metric("Description Similarity", f"{match['sim_desc']:.3f}")
                                st.metric("Title Similarity", f"{match['sim_title']:.3f}")
                            
                            with metrics_col2:
                                st.metric("Original Similarity", f"{match['sim_original']:.3f}")
                                st.metric("Adjusted Similarity", f"{match['sim_adjusted']:.3f}")
                            
                            with metrics_col3:
                                st.metric("Level Bonus", f"{match['level_bonus']:.3f}")
                        
                        st.markdown("---")

if __name__ == "__main__":
    main()
