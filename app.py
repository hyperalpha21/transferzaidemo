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
.info-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.success-box {
    background-color: #f0fff0;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #32cd32;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff8dc;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffa500;
    margin: 1rem 0;
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
        self.csv_url = "wm_courses_2025.csv"  # For local file fallback

    @st.cache_resource
    def load_model(_self):
        """Load the sentence transformer model"""
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    def extract_course_level(self, course_code):
        """Extract course level from course code"""
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
        """Calculate bonus for courses at or near the target level"""
        if course_level is None or target_level is None:
            return 0.0

        level_diff = abs(course_level - target_level)
        if level_diff == 0:
            return 0.15  # Perfect match
        elif level_diff == 100:
            return 0.12  # Very close match
        elif level_diff == 200:
            return 0.02  # Moderate difference
        else:
            return 0.0  # Too far away

    def load_csv_data(self, csv_source):
        """Load CSV data from file upload or local file"""
        try:
            if csv_source == "url":
                # Load from local file in the same directory
                if Path("wm_courses_2025.csv").exists():
                    df = pd.read_csv("wm_courses_2025.csv")
                else:
                    st.error("CSV file not found in app directory")
                    return None
            else:
                # Load from uploaded file
                df = pd.read_csv(csv_source)

            # Validate required columns
            required_cols = ['course_code', 'course_title', 'course_description']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return None

            # Clean data
            df.dropna(subset=['course_title', 'course_description'], inplace=True)
            df['course_level'] = df['course_code'].apply(self.extract_course_level)

            return df
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None

    def generate_embeddings(self, df, model):
        """Generate embeddings for course data"""
        try:
            course_texts = []
            for _, row in df.iterrows():
                combined_text = f"{row['course_code']} {row['course_title']} {row['course_description']}"
                course_texts.append(combined_text)

            # Check for cached embeddings
            embedding_path = Path("wm_embeddings_cached.pkl")
            if embedding_path.exists():
                with open(embedding_path, 'rb') as f:
                    embeddings = pickle.load(f)
                if len(embeddings) == len(course_texts):
                    return embeddings

            # Generate new embeddings
            progress_bar = st.progress(0)
            embeddings = model.encode(course_texts, batch_size=32, show_progress_bar=False)
            progress_bar.progress(100)

            # Cache embeddings
            with open(embedding_path, 'wb') as f:
                pickle.dump(embeddings, f)

            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return None

    def keyword_filter(self, courses_df, keywords):
        """Filter courses based on keywords"""
        if not keywords.strip():
            return courses_df

        keyword_list = [kw.strip().lower() for kw in keywords.split(',')]

        def contains_keywords(row):
            text = f"{row['course_code']} {row['course_title']} {row['course_description']}".lower()
            return any(keyword in text for keyword in keyword_list)

        filtered_df = courses_df[courses_df.apply(contains_keywords, axis=1)]
        return filtered_df

    def find_matches(self, external_courses, model, df, embeddings):
        """Find matches for external courses"""
        matches = {}

        for i, course in enumerate(external_courses):
            title = course['title']
            description = course['description']
            keywords = course['keywords']
            target_level = course['target_level']

            if not title or not description:
                continue

            # Apply keyword filter
            filtered_df = self.keyword_filter(df, keywords)
            if len(filtered_df) == 0:
                continue

            # Get embeddings for filtered courses
            if len(filtered_df) < len(df):
                filtered_texts = []
                for _, row in filtered_df.iterrows():
                    combined_text = f"{row['course_code']} {row['course_title']} {row['course_description']}"
                    filtered_texts.append(combined_text)
                filtered_embeddings = model.encode(filtered_texts)
            else:
                filtered_embeddings = embeddings

            # Generate embedding for external course
            external_text = f"{title} {description}"
            external_embedding = model.encode([external_text])

            # Calculate similarities
            similarities = cosine_similarity(external_embedding, filtered_embeddings)[0]

            # Apply level bonus
            if target_level:
                for idx, sim in enumerate(similarities):
                    course_level = filtered_df.iloc[idx]['course_level']
                    level_bonus = self.calculate_level_bonus(course_level, target_level)
                    similarities[idx] += level_bonus

            # Get top 5 matches
            top_5_indices = np.argsort(similarities)[-5:][::-1]

            course_matches = []
            for idx in top_5_indices:
                course_row = filtered_df.iloc[idx]
                original_similarity = cosine_similarity(external_embedding, [filtered_embeddings[idx]])[0][0]
                adjusted_similarity = similarities[idx]
                level_bonus = adjusted_similarity - original_similarity

                course_matches.append({
                    'course_code': course_row['course_code'],
                    'title': course_row['course_title'],
                    'description': course_row['course_description'],
                    'course_level': course_row['course_level'],
                    'similarity': original_similarity,
                    'adjusted_similarity': adjusted_similarity,
                    'level_bonus': level_bonus
                })

            matches[i] = course_matches

        return matches

    def calculate_transferability(self, title1, desc1, title2, desc2, model):
        """Calculate transferability score"""
        try:
            # Description similarity
            embeddings_desc = model.encode([desc1, desc2])
            similarity_desc = cosine_similarity([embeddings_desc[0]], [embeddings_desc[1]])[0][0]

            # Title similarity
            embeddings_title = model.encode([title1, title2])
            similarity_title = cosine_similarity([embeddings_title[0]], [embeddings_title[1]])[0][0]

            # Combined score
            combined_score = 1 / (1 + math.exp(-(-7.144 + 9.219 * similarity_desc + 5.141 * similarity_title)))

            return similarity_desc, similarity_title, combined_score
        except Exception:
            return None, None, None

    def get_transferability_category(self, score):
        """Get transferability category and emoji"""
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


def main():
    st.markdown('<h1 class="main-header">🎓 Course Transferability Analyzer</h1>', unsafe_allow_html=True)

    analyzer = CourseTransferabilityAnalyzer()

    # Sidebar for instructions toggle
    with st.sidebar:
        st.title("📋 App Controls")

        # Instructions toggle
        if st.button("Toggle Instructions"):
            st.session_state.instructions_visible = not st.session_state.instructions_visible

        # Model loading
        if st.session_state.model is None:
            if st.button("🔄 Load AI Model"):
                with st.spinner("Loading SentenceTransformer model..."):
                    st.session_state.model = analyzer.load_model()
                if st.session_state.model:
                    st.success("✅ Model loaded successfully!")
                else:
                    st.success("✅ Model loaded")

        # CSV source selection
        st.subheader("📁 Data Source")
        csv_source = st.radio(
            "Choose CSV source:",
            ["Upload File", "Use App's CSV File"],
            key="csv_source"
        )

        # Clear session button
        if st.button("🔄 Clear Session"):
            for key in list(st.session_state.keys()):
                if key != 'instructions_visible':
                    del st.session_state[key]
            st.experimental_rerun()

    # Instructions
    if st.session_state.instructions_visible:
        with st.expander("📋 How to Use This Tool", expanded=True):
            st.markdown("""
            ### Step-by-Step Instructions:

            1. **Load AI Model**: Click "Load AI Model" in the sidebar (required first step)
            2. **Load Course Data**: Choose to upload a CSV file or load from GitHub
            3. **Add External Courses**: Enter details for courses you want to find matches for
            4. **Find Matches**: Get the top 5 most similar courses with level preferences
            5. **Analyze Transferability**: Run detailed analysis on selected matches
            6. **Export Results**: Download your analysis as a CSV file

            ### Features:
            - **Level Filtering**: Target specific course levels (100-400)
            - **Keyword Filtering**: Add keywords to narrow search results
            - **Level Bonus System**: Courses at target levels get similarity bonuses
            - **Transferability Scoring**: Advanced algorithm for transfer likelihood

            ### Course Levels:
            - **100 Level**: Introductory courses
            - **200 Level**: Intermediate courses
            - **300 Level**: Advanced courses
            - **400 Level**: Senior-level courses
            """)

    # Main content
    if st.session_state.model is None:
        st.markdown('<div class="warning-box">⚠️ Please load the AI model first using the sidebar.</div>',
                    unsafe_allow_html=True)
        return

    # Step 1: Load CSV Data
    st.markdown('<h2 class="section-header">📁 Step 1: Load Course Catalog</h2>', unsafe_allow_html=True)

    csv_file = None
    if csv_source == "Upload File":
        csv_file = st.file_uploader("Upload CSV file", type=['csv'])
    else:
        st.info("Using app's built-in CSV file")
        csv_file = "url"

    if csv_file and st.button("Load Course Data"):
        with st.spinner("Loading course data..."):
            df = analyzer.load_csv_data(csv_file)
        if df is not None:
            st.session_state.university_courses_df = df

            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                embeddings = analyzer.generate_embeddings(df, st.session_state.model)
            if embeddings is not None:
                st.session_state.university_embeddings = embeddings

            st.markdown('<div class="success-box">✅ Course data loaded successfully!</div>', unsafe_allow_html=True)

            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Courses", len(df))
            with col2:
                levels = df['course_level'].value_counts().sort_index()
                st.metric("Course Levels", len(levels))
            with col3:
                st.metric("Embeddings", len(embeddings))

            # Show sample courses
            st.subheader("📊 Sample Courses")
            sample_df = df[['course_code', 'course_title', 'course_level']].head(5)
            st.dataframe(sample_df)

    # Step 2: Add External Courses
    if st.session_state.university_courses_df is not None:
        st.markdown('<h2 class="section-header">📚 Step 2: Add External Courses</h2>', unsafe_allow_html=True)

        num_courses = st.slider("Number of external courses:", 1, 10, 3)

        external_courses = []
        for i in range(num_courses):
            with st.expander(f"🔍 External Course {i + 1}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    title = st.text_input(f"Course Title {i + 1}", key=f"title_{i}")
                    keywords = st.text_input(f"Keywords {i + 1} (optional)", key=f"keywords_{i}")

                with col2:
                    target_level = st.selectbox(
                        f"Target Level {i + 1}",
                        [None, 100, 200, 300, 400],
                        format_func=lambda x: "Any Level" if x is None else f"{x} Level",
                        key=f"level_{i}"
                    )

                description = st.text_area(f"Course Description {i + 1}", key=f"desc_{i}")

                if title and description:
                    external_courses.append({
                        'title': title,
                        'description': description,
                        'keywords': keywords,
                        'target_level': target_level
                    })

        # Step 3: Find Matches
        if external_courses:
            st.markdown('<h2 class="section-header">🔍 Step 3: Find Matches</h2>', unsafe_allow_html=True)

            if st.button("Find Top 5 Matches"):
                with st.spinner("Finding matches..."):
                    matches = analyzer.find_matches(
                        external_courses,
                        st.session_state.model,
                        st.session_state.university_courses_df,
                        st.session_state.university_embeddings
                    )
                st.session_state.course_matches = matches

                # Display matches
                for i, course_matches in matches.items():
                    st.subheader(f"📚 External Course {i + 1}: {external_courses[i]['title']}")

                    match_data = []
                    for j, match in enumerate(course_matches):
                        level_str = f"Level {match['course_level']}" if match['course_level'] else "Level Unknown"
                        match_data.append({
                            'Rank': j + 1,
                            'Course Code': match['course_code'],
                            'Title': match['title'],
                            'Level': level_str,
                            'Similarity': f"{match['similarity']:.4f}",
                            'Level Bonus': f"{match['level_bonus']:.4f}",
                            'Adjusted Score': f"{match['adjusted_similarity']:.4f}"
                        })

                    st.dataframe(pd.DataFrame(match_data))

    # Step 4: Analyze Transferability
    if st.session_state.course_matches:
        st.markdown('<h2 class="section-header">🎯 Step 4: Analyze Transferability</h2>', unsafe_allow_html=True)

        # Selection interface
        selected_matches = {}
        for i, course_matches in st.session_state.course_matches.items():
            st.subheader(f"Select matches for External Course {i + 1}")

            options = []
            for j, match in enumerate(course_matches):
                level_str = f"(L{match['course_level']})" if match['course_level'] else ""
                label = f"{match['course_code']}: {match['title']} {level_str} - {match['adjusted_similarity']:.3f}"
                options.append(label)

            selected = st.multiselect(
                f"Choose courses to analyze:",
                options,
                key=f"select_{i}"
            )

            if selected:
                selected_matches[i] = [options.index(sel) for sel in selected]

        if selected_matches and st.button("Analyze Selected Matches"):
            with st.spinner("Analyzing transferability..."):
                results = []

                for course_idx, match_indices in selected_matches.items():
                    external_course = external_courses[course_idx]
                    course_matches = st.session_state.course_matches[course_idx]

                    for match_idx in match_indices:
                        match = course_matches[match_idx]

                        # Calculate transferability
                        desc_sim, title_sim, combined_score = analyzer.calculate_transferability(
                            external_course['title'],
                            external_course['description'],
                            match['title'],
                            match['description'],
                            st.session_state.model
                        )

                        if combined_score is not None:
                            category, emoji = analyzer.get_transferability_category(combined_score)

                            result = {
                                'External_Course_Number': course_idx + 1,
                                'External_Course_Title': external_course['title'],
                                'External_Course_Description': external_course['description'],
                                'Target_Level': external_course['target_level'],
                                'WM_Course_Code': match['course_code'],
                                'WM_Course_Title': match['title'],
                                'WM_Course_Description': match['description'],
                                'WM_Course_Level': match['course_level'],
                                'Initial_Similarity': round(match['similarity'], 4),
                                'Level_Bonus': round(match['level_bonus'], 4),
                                'Adjusted_Similarity': round(match['adjusted_similarity'], 4),
                                'Description_Similarity': round(desc_sim, 4),
                                'Title_Similarity': round(title_sim, 4),
                                'Combined_Transferability_Score': round(combined_score, 4),
                                'Transferability_Category': category,
                                'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }

                            results.append(result)

                st.session_state.all_comparisons.extend(results)

                # Display results
                if results:
                    st.markdown('<div class="success-box">✅ Analysis complete!</div>', unsafe_allow_html=True)

                    # Summary statistics
                    scores = [r['Combined_Transferability_Score'] for r in results]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Analyzed Pairs", len(results))
                    with col2:
                        st.metric("Avg Score", f"{np.mean(scores):.4f}")
                    with col3:
                        st.metric("Max Score", f"{np.max(scores):.4f}")
                    with col4:
                        st.metric("Min Score", f"{np.min(scores):.4f}")

                    # Detailed results
                    st.subheader("📊 Detailed Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)

                    # Export option
                    if st.button("📥 Export Results to CSV"):
                        csv_data = results_df.to_csv(index=False)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f'transferability_analysis_{timestamp}.csv'

                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv"
                        )


if __name__ == "__main__":
    main()
