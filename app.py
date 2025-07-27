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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');

/* Modern, clean aesthetic inspired by top UI software */
.main-header { 
    font-family: 'SF Pro Display', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 3.5rem; 
    font-weight: 300;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center; 
    margin-bottom: 3rem;
    letter-spacing: -0.02em;
}

.step-header { 
    font-family: 'Inter', sans-serif;
    font-size: 1.75rem; 
    color: #1a1a1a; 
    margin: 3rem 0 1.5rem 0; 
    padding: 0;
    font-weight: 600;
    letter-spacing: -0.01em;
}

.modern-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 32px;
    margin: 24px 0;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.04);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.modern-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.12);
}

.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 16px 32px;
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.24);
    letter-spacing: -0.01em;
}

.primary-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.32);
}

.secondary-button {
    background: #f8f9fa;
    color: #495057;
    border: 1px solid #e9ecef;
    padding: 12px 24px;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.secondary-button:hover {
    background: #e9ecef;
    transform: translateY(-1px);
}

.metric-display {
    background: #f8f9ff;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    border: 1px solid #e6e8ff;
}

.metric-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
    font-family: 'SF Pro Display', sans-serif;
    line-height: 1;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 14px;
    color: #6c757d;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.transfer-result-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 28px;
    margin: 20px 0;
    box-shadow: 0 2px 16px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.06);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.transfer-result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
}

.transfer-result-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.course-code {
    background: #f1f3f9;
    color: #495057;
    padding: 6px 12px;
    border-radius: 8px;
    font-family: 'SF Pro Display', monospace;
    font-weight: 600;
    font-size: 14px;
    display: inline-block;
}

.percentage-display {
    font-family: 'SF Pro Display', sans-serif;
    font-weight: 700;
    font-size: 3rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.help-container {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.04) 0%, rgba(118, 75, 162, 0.04) 100%);
    border: 1px solid rgba(102, 126, 234, 0.08);
    border-radius: 20px;
    padding: 32px;
    margin: 32px 0;
}

.help-container h3 {
    font-family: 'SF Pro Display', sans-serif;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 24px;
    font-size: 1.5rem;
}

.help-container ol {
    font-family: 'Inter', sans-serif;
    line-height: 1.7;
    color: #495057;
}

.help-container li {
    margin-bottom: 12px;
}

.status-badge {
    background: #d4edda;
    color: #155724;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.upload-area {
    border: 2px dashed #dee2e6;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    background: #f8f9fa;
    transition: all 0.3s ease;
    margin: 20px 0;
}

.upload-area:hover {
    border-color: #667eea;
    background: #f8f9ff;
}

/* Category specific colors */
.very-high { color: #28a745; }
.likely { color: #17a2b8; }
.possible { color: #ffc107; }
.unlikely { color: #fd7e14; }
.low { color: #dc3545; }

/* Smooth transitions for all interactive elements */
* {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
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
        # Use the original W&M catalog loading logic
        path = 'wm_courses_2025.csv'
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                continue
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
    
    st.markdown('<h1 class="main-header">🎓 TransferzAI</h1>', unsafe_allow_html=True)

    # Help instructions with inline buttons
    if st.session_state.show_help:
        st.markdown("""
        <div class='help-container'>
        <h3>How to Use TransferzAI</h3>
        <ol>
            <li><strong>Start the Model:</strong> Click the button below to load the AI model (required for analysis)</li>
            <li><strong>Load Course Catalog:</strong> Choose your data source and load the catalog</li>
            <li><strong>Add External Courses:</strong> Input course details with title, description, and optional filters</li>
            <li><strong>Analyze:</strong> Run the AI analysis to find matching courses with transferability scores</li>
        </ol>
        <p><strong>💡 CSV Format:</strong> Must contain columns: <span class="course-code">course_code</span>, <span class="course-code">course_title</span>, <span class="course-code">course_description</span></p>
        </div>
        """, unsafe_allow_html=True)

    # Model loading section
    if not st.session_state.model:
        st.markdown('<div class="modern-card">')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🤖 AI Model Setup")
            st.markdown("Initialize the AI model to begin course analysis")
            if st.button("Start AI Model", key="start_model", help="Load the AI model for course analysis"):
                with st.spinner("Loading AI model..."):
                    st.session_state.model = load_model()
                    if st.session_state.model:
                        st.success("✅ Model loaded successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to load model")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="status-badge">✅ AI Model Ready</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📋 Control Panel")
        
        if st.button("ℹ️ Show/Hide Help"):
            st.session_state.show_help = not st.session_state.show_help
        
        st.markdown("---")
        
        if st.button("🔄 Reset All"):
            for key in ['model', 'courses_df', 'courses_emb', 'matches', 'external_courses']:
                if key == 'matches' or key == 'external_courses':
                    st.session_state[key] = {} if key == 'matches' else []
                else:
                    st.session_state[key] = None
            st.rerun()

    # Step 1: Load catalog
    st.markdown('<div class="step-header">📁 Load Course Catalog</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="modern-card">')
    # Course catalog selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        src = st.radio("**Choose Data Source**", ["Upload CSV", "W&M Catalog"], key="src", horizontal=True)
    
    file_to_process = None
    if src == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your course catalog CSV file")
        if uploaded_file:
            file_to_process = uploaded_file
            st.success(f"📄 File uploaded: {uploaded_file.name}")
    else:
        file_to_process = "url"
        st.info("📊 Using William & Mary Course Catalog")
    
    # Centered load button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if file_to_process and st.session_state.model:
            if st.button("Load Catalog", key="load_catalog", help="Load and process the course catalog"):
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
        elif not st.session_state.model:
            st.info("⚠️ Please start the AI model first")
    st.markdown('</div>', unsafe_allow_html=True)

    # Show catalog info if loaded
    if st.session_state.courses_df is not None:
        with st.expander("📊 Catalog Overview", expanded=False):
            df = st.session_state.courses_df
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-display">')
                st.markdown(f'<div class="metric-number">{len(df)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Courses</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-display">')
                departments = df['course_code'].str[:4].nunique() if 'course_code' in df.columns else 0
                st.markdown(f'<div class="metric-number">{departments}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Departments</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-display">')
                levels = df['level'].nunique() if 'level' in df.columns else 0
                st.markdown(f'<div class="metric-number">{levels}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Course Levels</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("**Sample Courses:**")
            st.dataframe(df[['course_code', 'course_title']].head(), use_container_width=True)

    # Step 2: Add external courses
    external_courses = []
    if st.session_state.courses_df is not None and st.session_state.model is not None:
        st.markdown('<div class="step-header">📚 Add Your External Courses</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="modern-card">')
        num_courses = st.slider("Number of courses to analyze", 1, 10, 3, help="Select how many external courses you want to analyze")
        
        for i in range(num_courses):
            with st.expander(f"📝 Course {i+1}", expanded=(i < 2)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    title = st.text_input("Course Title", key=f"title_{i}", 
                                        placeholder="e.g., Introduction to Computer Science",
                                        help="Enter the full title of your external course")
                    description = st.text_area("Course Description", key=f"desc_{i}", height=120,
                                             placeholder="Provide a detailed description of the course content, topics covered, and learning objectives...",
                                             help="The more detailed the description, the better the AI can match it")
                
                with col2:
                    keywords = st.text_input("Keywords (optional)", key=f"keywords_{i}",
                                           placeholder="math, science, programming",
                                           help="Add keywords to help filter matching courses")
                    level = st.selectbox("Target Level", [None, 100, 200, 300, 400], key=f"level_{i}",
                                       format_func=lambda x: "Any Level" if x is None else f"{x}-level",
                                       help="Course level to prioritize in matching (100=freshman, 400=senior)")
                
                if title and description:
                    external_courses.append({
                        'title': title.strip(),
                        'description': description.strip(),
                        'keywords': keywords.strip() if keywords else '',
                        'target_level': level
                    })
        st.markdown('</div>', unsafe_allow_html=True)

        # Store external courses in session state
        st.session_state.external_courses = external_courses

        # Step 3: Analyze
        if external_courses:
            st.markdown('<div class="step-header">🔍 Run AI Analysis</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="modern-card">')
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### Ready to Analyze")
                st.markdown(f"Analyzing **{len(external_courses)}** external course{'s' if len(external_courses) != 1 else ''} for transferability")
                if st.button("Start Analysis", key="analyze_button", help="Run AI analysis to find transferable courses"):
                    with st.spinner("🤖 AI is analyzing course transferability..."):
                        matches = find_matches_with_logit(
                            external_courses,
                            st.session_state.model,
                            st.session_state.courses_df,
                            st.session_state.courses_emb
                        )
                        st.session_state.matches = matches
                        if matches:
                            st.success("✅ Analysis complete!")
                            st.balloons()
                        else:
                            st.warning("⚠️ No matches found. Try adjusting your course descriptions or keywords.")
            st.markdown('</div>', unsafe_allow_html=True) Courses", type="primary", help="Run AI analysis to find transferable courses"):
                    with st.spinner("🤖 AI is analyzing course transferability..."):
                        matches = find_matches_with_logit(
                            external_courses,
                            st.session_state.model,
                            st.session_state.courses_df,
                            st.session_state.courses_emb
                        )
                        st.session_state.matches = matches
                        if matches:
                            st.success("✅ Analysis complete!")
                            st.balloons()
                        else:
                            st.warning("⚠️ No matches found. Try adjusting your course descriptions or keywords.")

    # Step 4: Display Results
    if st.session_state.matches:
        st.markdown('<div class="step-header">🎯 Step 4: Transfer Analysis Results</div>', unsafe_allow_html=True)
        st.markdown("*Percentage represents the probability that your external course will transfer as equivalent to the matched course*")
        
        for idx, matches in st.session_state.matches.items():
            if idx < len(st.session_state.external_courses):
                external_course = st.session_state.external_courses[idx]
                
                st.markdown(f"### 📘 External Course {idx+1}: {external_course['title']}")
                
                if not matches:
                    st.warning("No suitable matches found for this course.")
                    continue
                
                # Display matches with enhanced styling
                for rank, match in enumerate(matches, 1):
                    pct = round(match['transfer_score'] * 100, 1)
                    
                    # Create styled result container
                    st.markdown(f"""
                    <div class="transfer-result">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                            <div>
                                <h4 style="margin: 0; color: #64ffda;">#{rank} {match['emoji']} {match['category']}</h4>
                                <p style="margin: 4px 0 0 0; color: #888;">
                                    <span class="percentage-text">{pct}%</span> chance of transferring as this class
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{pct}%</div>
                            </div>
                        </div>
                        <div>
                            <p style="margin: 8px 0; font-size: 1.1em;">
                                <span class="code-text">{match['code']}</span>: <strong>{match['title']}</strong>
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed metrics in expander
                    with st.expander("📊 Detailed Analysis Metrics", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Description Similarity", 
                                f"{match['sim_desc']:.3f}",
                                help="How similar the course descriptions are (0-1 scale)"
                            )
                            st.metric(
                                "Title Similarity", 
                                f"{match['sim_title']:.3f}",
                                help="How similar the course titles are (0-1 scale)"
                            )
                        
                        with col2:
                            st.metric(
                                "Base Similarity", 
                                f"{match['sim_original']:.3f}",
                                help="Raw similarity score before adjustments"
                            )
                            st.metric(
                                "Adjusted Similarity", 
                                f"{match['sim_adjusted']:.3f}",
                                help="Similarity score after level and keyword adjustments"
                            )
                        
                        with col3:
                            st.metric(
                                "Level Bonus", 
                                f"{match['level_bonus']:.3f}",
                                help="Bonus points for matching course levels (100, 200, 300, 400)"
                            )
                            st.metric(
                                "AI Transfer Score", 
                                f"{match['transfer_score']:.3f}",
                                help="Final AI-computed probability of successful transfer (0-1 scale)"
                            )
                
                st.markdown("---")

if __name__ == "__main__":
    main()
