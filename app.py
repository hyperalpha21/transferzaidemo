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
    page_title="Will My Courses Transfer?",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - simplified and cleaner
st.markdown("""
<style>
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
    margin-bottom: 1rem;
    padding: 10px;
    background-color: #f0f8f0;
    border-radius: 8px;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.result-bar {
    height: 20px;
    background: #eee;
    border-radius: 10px;
    overflow: hidden;
    margin: 15px 0;
}
.result-bar-fill {
    height: 100%;
    transition: width 0.5s;
    border-radius: 10px;
}
.help-text {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #17a2b8;
    margin: 10px 0;
    color: #333333 !important;
}
.help-text h3 {
    color: #2c3e50 !important;
    margin-bottom: 10px;
}
.help-text ol {
    color: #333333 !important;
}
.help-text li {
    color: #333333 !important;
    margin-bottom: 5px;
}
.help-text strong {
    color: #2c3e50 !important;
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
if 'course_matches' not in st.session_state:
    st.session_state.course_matches = {}
if 'show_help' not in st.session_state:
    st.session_state.show_help = True

class CourseTransferChecker:
    def __init__(self):
        self.csv_url = "wm_courses_2025.csv"

    @st.cache_resource
    def load_model(_self):
        """Load the AI model that compares courses"""
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            return model
        except Exception as e:
            st.error(f"Could not load the comparison tool: {str(e)}")
            return None

    def extract_course_level(self, course_code):
        """Figure out if a course is intro (100), intermediate (200), advanced (300), etc."""
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
        """Give extra points for courses at the same level"""
        if course_level is None or target_level is None:
            return 0.0
        diff = abs(course_level - target_level)
        if diff == 0:
            return 0.15  # Same level = big bonus
        elif diff == 100:
            return 0.12  # One level off = smaller bonus
        elif diff == 200:
            return 0.02  # Two levels off = tiny bonus
        return 0.0

    @st.cache_data
    def load_csv_data(_self, csv_source):
        """Load the university course catalog with better encoding handling"""
        try:
            # Try different encodings to handle various CSV files
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    if csv_source == "url":
                        if Path("wm_courses_2025.csv").exists():
                            df = pd.read_csv("wm_courses_2025.csv", encoding=encoding)
                        else:
                            st.error("Course catalog file not found. Please upload your own CSV file.")
                            return None
                    else:
                        df = pd.read_csv(csv_source, encoding=encoding)
                    
                    # If we get here, the encoding worked
                    st.success(f"✅ Successfully loaded CSV file (encoding: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error("Could not read the CSV file. The file might be corrupted or in an unsupported format.")
                return None

            # Check for required columns
            required = ['course_code', 'course_title', 'course_description']
            missing_cols = [col for col in required if col not in df.columns]
            
            if missing_cols:
                st.error(f"Your CSV file is missing these required columns: {', '.join(missing_cols)}")
                st.info("Make sure your CSV has columns named: course_code, course_title, course_description")
                return None

            # Clean up the data
            original_count = len(df)
            df.dropna(subset=['course_title', 'course_description'], inplace=True)
            
            if len(df) < original_count:
                st.info(f"Removed {original_count - len(df)} courses with missing information")
            
            df['course_level'] = df['course_code'].apply(_self.extract_course_level)
            
            st.info(f"📚 Loaded {len(df)} courses from the catalog")
            return df
            
        except Exception as e:
            st.error(f"Error loading course catalog: {str(e)}")
            return None

    @st.cache_data
    def generate_embeddings(_self, df, _model):
        """Create numerical representations of courses for comparison"""
        try:
            texts = [f"{row['course_code']} {row['course_title']} {row['course_description']}" 
                    for _, row in df.iterrows()]
            
            # Create a hash of the course data to use as cache key
            import hashlib
            data_hash = hashlib.md5(str(sorted(texts)).encode()).hexdigest()
            cache_path = Path(f"embeddings_cache_{data_hash}.pkl")
            
            # Try to use cached version first
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_embeddings = pickle.load(f)
                    if len(cached_embeddings) == len(texts):
                        st.info("✅ Using previously processed course data (much faster!)")
                        return cached_embeddings
                except:
                    pass  # If cache fails, just generate new ones
            
            # Generate new embeddings with progress
            st.info("🔄 Processing course descriptions for comparison...")
            progress_bar = st.progress(0)
            
            # Process in smaller batches for better progress tracking
            batch_size = 16  # Smaller batches for better progress updates
            total_batches = (len(texts) + batch_size - 1) // batch_size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = _model.encode(batch, batch_size=batch_size, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                current_batch = (i // batch_size) + 1
                progress_bar.progress(current_batch / total_batches)
            
            progress_bar.empty()
            embeddings = np.array(all_embeddings)
            
            # Save to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                st.success("✅ Course data processed and saved for future use!")
            except:
                st.warning("Could not save processed data to cache, but continuing...")
                
            return embeddings
            
        except Exception as e:
            st.error(f"Error processing course descriptions: {str(e)}")
            return None

    def find_matches(self, external_courses, model, df, embeddings):
        """Find the best matching courses from the university catalog"""
        matches = {}
        
        total_courses = len(external_courses)
        progress_bar = st.progress(0)
        st.info("🔍 Searching for similar courses...")
        
        for i, course in enumerate(external_courses):
            title = course['title']
            description = course['description']
            keywords = course['keywords']
            target_level = course['target_level']
            
            if not title or not description:
                continue
            
            # Filter by keywords if provided
            filtered_df = df.copy()
            filtered_embeddings = embeddings
            
            if keywords.strip():
                keyword_list = [k.strip().lower() for k in keywords.split(',')]
                mask = df.apply(
                    lambda row: any(
                        keyword in f"{row['course_code']} {row['course_title']} {row['course_description']}".lower() 
                        for keyword in keyword_list
                    ), axis=1
                )
                filtered_df = df[mask]
                filtered_embeddings = embeddings[mask.values]
                
                if len(filtered_df) == 0:
                    st.warning(f"No courses found matching keywords: {keywords}")
                    continue
            
            # Create embedding for external course (cached if possible)
            external_text = f"{title} {description}"
            external_embedding = model.encode([external_text])
            
            # Calculate similarities using vectorized operations
            similarities = cosine_similarity(external_embedding, filtered_embeddings)[0]
            
            # Add level bonus efficiently
            if target_level:
                level_bonuses = filtered_df['course_level'].apply(
                    lambda x: self.calculate_level_bonus(x, target_level)
                ).values
                similarities += level_bonuses
            
            # Get top 5 matches efficiently
            top_indices = np.argpartition(similarities, -5)[-5:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
            course_matches = []
            for idx in top_indices:
                row = filtered_df.iloc[idx]
                original_similarity = cosine_similarity(external_embedding, [filtered_embeddings[idx]])[0][0]
                adjusted_similarity = similarities[idx]
                level_bonus = adjusted_similarity - original_similarity
                
                course_matches.append({
                    'course_code': row['course_code'],
                    'title': row['course_title'],
                    'description': row['course_description'],
                    'course_level': row['course_level'],
                    'similarity': original_similarity,
                    'adjusted_similarity': adjusted_similarity,
                    'level_bonus': level_bonus
                })
            
            matches[i] = course_matches
            
            # Update progress
            progress_bar.progress((i + 1) / total_courses)
        
        progress_bar.empty()
        return matches

    def calculate_transferability(self, title1, desc1, title2, desc2, model):
        """Calculate how likely a course is to transfer"""
        try:
            # Compare course descriptions
            desc_embeddings = model.encode([desc1, desc2])
            desc_similarity = cosine_similarity([desc_embeddings[0]], [desc_embeddings[1]])[0][0]
            
            # Compare course titles
            title_embeddings = model.encode([title1, title2])
            title_similarity = cosine_similarity([title_embeddings[0]], [title_embeddings[1]])[0][0]
            
            # Calculate final transferability score using a trained formula
            score = 1 / (1 + math.exp(-(-7.144 + 9.219 * desc_similarity + 5.141 * title_similarity)))
            
            return desc_similarity, title_similarity, score
            
        except Exception as e:
            st.error(f"Error calculating transferability: {str(e)}")
            return None, None, None

    def get_transferability_category(self, score):
        """Convert score to easy-to-understand category"""
        if score >= 0.85:
            return "Very Likely to Transfer", "🟢"
        elif score >= 0.73:
            return "Likely to Transfer", "🔵"
        elif score >= 0.6:
            return "Might Transfer", "🟡"
        elif score >= 0.4:
            return "Probably Won't Transfer", "🟠"
        else:
            return "Very Unlikely to Transfer", "🔴"

def main():
    st.markdown('<h1 class="main-header">🎓 Will My Courses Transfer?</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("📋 Menu")
        
        if st.button("ℹ️ Show/Hide Help"):
            st.session_state.show_help = not st.session_state.show_help
        
        st.markdown("---")
        
        # Model loading
        if st.session_state.model is None:
            if st.button("🚀 Start the App"):
                with st.spinner("Loading the course comparison tool..."):
                    analyzer = CourseTransferChecker()
                    st.session_state.model = analyzer.load_model()
                if st.session_state.model:
                    st.success("✅ Ready to go!")
                else:
                    st.error("❌ Something went wrong")
        else:
            st.success("✅ App is ready!")
        
        st.markdown("---")
        
        # Data source selection
        st.subheader("📁 Course Catalog")
        csv_source = st.radio(
            "Where are your university courses?",
            ["Upload a file", "Use built-in catalog"],
            key="csv_source"
        )
        
        st.markdown("---")
        
        if st.button("🔄 Start Over"):
            for key in list(st.session_state.keys()):
                if key != 'show_help':
                    del st.session_state[key]
            st.rerun()

    # Help section
    if st.session_state.show_help:
        st.markdown("""
        <div class="help-text">
        <h3>How to Use This App:</h3>
        <ol>
        <li><strong>Start the App:</strong> Click "Start the App" in the sidebar</li>
        <li><strong>Load Courses:</strong> Upload your university's course catalog or use the built-in one</li>
        <li><strong>Add Your Courses:</strong> Enter the courses you want to transfer</li>
        <li><strong>Find Matches:</strong> The app will find similar courses at the new university</li>
        <li><strong>Check Transferability:</strong> See how likely each course is to transfer</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    # Check if model is loaded
    if st.session_state.model is None:
        st.warning("⚠️ Please click 'Start the App' in the sidebar to begin!")
        return

    analyzer = CourseTransferChecker()

    # Step 1: Load Course Catalog
    st.markdown('<div class="step-header">📁 Step 1: Load University Course Catalog</div>', unsafe_allow_html=True)
    
    if csv_source == "Upload a file":
        st.info("Upload a CSV file with columns: course_code, course_title, course_description")
        csv_file = st.file_uploader("Choose your CSV file", type=['csv'])
    else:
        csv_file = "url"
        st.info("Using the built-in William & Mary course catalog")

    if csv_file and st.button("📂 Load Courses"):
        with st.spinner("Loading course catalog..."):
            df = analyzer.load_csv_data(csv_file)
            
        if df is not None:
            st.session_state.university_courses_df = df
            
            with st.spinner("Preparing courses for comparison..."):
                embeddings = analyzer.generate_embeddings(df, st.session_state.model)
                
            if embeddings is not None:
                st.session_state.university_embeddings = embeddings
                st.success("✅ Course catalog ready!")
                
                # Show preview
                with st.expander("📋 Preview of loaded courses"):
                    preview_df = df[['course_code', 'course_title', 'course_level']].head(10)
                    st.dataframe(preview_df, use_container_width=True)

    # Step 2: Add External Courses
    if st.session_state.university_courses_df is not None:
        st.markdown('<div class="step-header">📚 Step 2: Enter Your Courses</div>', unsafe_allow_html=True)
        
        st.info("Add the courses you took at your previous school that you want to transfer")
        
        num_courses = st.slider("How many courses do you want to check?", 1, 10, 3)
        
        external_courses = []
        
        for i in range(num_courses):
            with st.expander(f"Course #{i+1}", expanded=(i < 2)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    title = st.text_input(f"Course Title", key=f"title_{i}", 
                                        placeholder="e.g., Introduction to Psychology")
                    description = st.text_area(f"Course Description", key=f"desc_{i}",
                                             placeholder="Describe what you learned in this course...",
                                             height=100)
                
                with col2:
                    keywords = st.text_input(f"Keywords (optional)", key=f"keywords_{i}",
                                           placeholder="psychology, behavior")
                    target_level = st.selectbox(
                        f"Course Level",
                        [None, 100, 200, 300, 400],
                        format_func=lambda x: "Any Level" if x is None else f"{x} Level",
                        key=f"level_{i}"
                    )
                
                if title and description:
                    external_courses.append({
                        'title': title,
                        'description': description,
                        'keywords': keywords,
                        'target_level': target_level
                    })

        # Step 3: Find Matches
        if external_courses:
            st.markdown('<div class="step-header">🔍 Step 3: Find Similar Courses</div>', unsafe_allow_html=True)
            
            if st.button("🔍 Find Matching Courses", type="primary"):
                with st.spinner("Searching for similar courses..."):
                    matches = analyzer.find_matches(
                        external_courses, 
                        st.session_state.model,
                        st.session_state.university_courses_df, 
                        st.session_state.university_embeddings
                    )
                    st.session_state.course_matches = matches
                
                if matches:
                    st.success(f"✅ Found matches for {len(matches)} courses!")

    # Step 4: Select and Analyze
    if st.session_state.course_matches:
        st.markdown('<div class="step-header">✅ Step 4: Select Courses to Analyze</div>', unsafe_allow_html=True)
        
        selected_matches = {}
        
        for course_idx, matches in st.session_state.course_matches.items():
            if course_idx < len(external_courses):
                ext_course = external_courses[course_idx]
                st.write(f"**Your Course:** {ext_course['title']}")
                
                # Create options for selection
                match_options = []
                for i, match in enumerate(matches):
                    similarity_pct = int(match['adjusted_similarity'] * 100)
                    option = f"{match['course_code']} - {match['title']} ({similarity_pct}% match)"
                    match_options.append(option)
                
                selected = st.multiselect(
                    "Select which courses you want to check for transfer:",
                    match_options,
                    key=f"select_{course_idx}"
                )
                
                if selected:
                    selected_indices = [match_options.index(sel) for sel in selected]
                    selected_matches[course_idx] = selected_indices
                
                st.markdown("---")

        # Step 5: Final Analysis
        if selected_matches:
            st.markdown('<div class="step-header">🎯 Step 5: Transfer Analysis</div>', unsafe_allow_html=True)
            
            if st.button("📊 Check Transfer Likelihood", type="primary"):
                results = []
                
                progress_bar = st.progress(0)
                total_analyses = sum(len(indices) for indices in selected_matches.values())
                current_analysis = 0
                
                for course_idx, match_indices in selected_matches.items():
                    external_course = external_courses[course_idx]
                    
                    for match_idx in match_indices:
                        match = st.session_state.course_matches[course_idx][match_idx]
                        
                        # Calculate transferability
                        desc_sim, title_sim, score = analyzer.calculate_transferability(
                            external_course['title'],
                            external_course['description'],
                            match['title'],
                            match['description'],
                            st.session_state.model
                        )
                        
                        if score is not None:
                            category, emoji = analyzer.get_transferability_category(score)
                            results.append({
                                'Your Course': external_course['title'],
                                'University Code': match['course_code'],
                                'University Course': match['title'],
                                'Transfer Score': score,
                                'Category': category,
                                'Emoji': emoji
                            })
                        
                        current_analysis += 1
                        progress_bar.progress(current_analysis / total_analyses)
                
                progress_bar.empty()
                
                if results:
                    st.success("✅ Analysis complete!")
                    
                    # Display results as cards
                    color_mapping = {
                        "Very Likely to Transfer": "#d4edda",
                        "Likely to Transfer": "#cce5ff", 
                        "Might Transfer": "#fff3cd",
                        "Probably Won't Transfer": "#ffe5b4",
                        "Very Unlikely to Transfer": "#f8d7da"
                    }
                    
                    for result in results:
                        bg_color = color_mapping[result['Category']]
                        score = result['Transfer Score']
                        
                        if score > 0.7:
                            bar_color = '#28a745'  # Green
                        elif score > 0.4:
                            bar_color = '#ffc107'  # Yellow
                        else:
                            bar_color = '#dc3545'  # Red
                        
                        st.markdown(f"""
                        <div class='result-card' style='background-color: {bg_color}; border: 1px solid #ddd;'>
                            <h2 style='text-align: center; margin-bottom: 15px;'>
                                {result['Emoji']} {result['Category']}
                            </h2>
                            <p style='text-align: center; font-size: 1.1em; margin-bottom: 15px;'>
                                <strong>{result['Your Course']}</strong><br>
                                ↓<br>
                                <strong>{result['University Code']}: {result['University Course']}</strong>
                            </p>
                            <div class='result-bar'>
                                <div class='result-bar-fill' style='width: {score*100}%; background-color: {bar_color};'></div>
                            </div>
                            <p style='text-align: center; font-size: 1.2em; font-weight: bold; margin-top: 10px;'>
                                Transfer Likelihood: {round(score*100, 1)}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Summary table
                    with st.expander("📊 Summary Table"):
                        summary_df = pd.DataFrame(results)
                        summary_df['Transfer Score'] = summary_df['Transfer Score'].round(3)
                        st.dataframe(summary_df.drop(['Emoji'], axis=1), use_container_width=True)

if __name__ == "__main__":
    main()
