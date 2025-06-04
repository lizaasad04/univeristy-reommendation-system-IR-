import streamlit as st
import pandas as pd
import plotly.express as px
from main import AustralianUniversitiesIR  # your IR system class

# Page configuration
st.set_page_config(
    page_title="Australian Universities Course Search",
    page_icon="🇦🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #003366;
    }
    .subheader {
        font-size: 1.5rem;
        color: #004080;
    }
    .result-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #004080;
    }
    .metric-card {
        background-color: #e6f2ff;
        border-radius: 5px;
        padding: 5px;
        text-align: center;
    }
    .stButton button {
        background-color: #004080;
        color: white;
    }
    .university-name {
        color: #004080;
        font-weight: bold;
    }
    .course-name {
        color: #004080;
    }
</style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown("<h1 class='main-header'>University Recommendation System</h1>", unsafe_allow_html=True)


# Load and prepare the IR system (cache to avoid reloading every time)
@st.cache_resource #streamlit func, stores result of a func, so func executes once, and after that, result is stored in cache
def load_ir_system():
    ir = AustralianUniversitiesIR("aus_uni.xlsx") #creating an instance for our class

    # Try to load cached data first
    if ir.load_cached_data():
        st.success(" ")
        return ir

    # If no cached data, load from Excel and save cache for next time
    st.info("ℹ No cached data found. Loading from Excel file (this might take a while)...")
    if ir.load_data() and ir.merge_datasets():
        # Save the processed data for future use
        ir.save_processed_data()
        return ir
    else:
        return None


#is a Streamlit feature that allows you to preserve variable values between reruns of your app.
#This is essential for maintaining state in a Streamlit application since Streamlit reruns the entire script on each interaction. `st.session_state`

# checks if recent searches exists in our session state, if not, initialises an empty list for it
#stores history of user queries
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []

if 'filtered_results' not in st.session_state:
    st.session_state.filtered_results = None

if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

if 'favorites' not in st.session_state:
    st.session_state.favorites = []


# Function to get unique values from DataFrame column
def get_unique_values(df, column):
    if df is not None and not df.empty and column in df.columns:
        return sorted(list(df[column].dropna().unique()))
    return []


# Load the IR system
with st.spinner("🔄 Loading search system..."):
    ir_system = load_ir_system()

# Error handling if IR system fails to load
if ir_system is None:
    st.error("❌ Failed to load or prepare the data. Please check your data source and try again.")
    st.stop()

# Sidebar for filters and options
with st.sidebar:
    st.markdown("<h2 class='subheader'>🔍 Search Options</h2>", unsafe_allow_html=True)

    # Advanced Search Options
    with st.expander("⚙ Advanced Search Settings", expanded=False):
        search_method = st.selectbox(
            "Search Method:",
            options=["Standard", "Semantic", "Fuzzy"],
            help="Standard: keyword matching, Semantic: meaning-based, Fuzzy: approximate matching"
        )

        top_n = st.slider(
            "Number of Results:",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )

    # Filters (will be populated after search)
    st.markdown("<h2 class='subheader'>🔍 Filter Results</h2>", unsafe_allow_html=True)

    st.session_state.filter_locations = st.multiselect(
        "Locations:",
        options=["All Locations"],
        default=["All Locations"],
        disabled=st.session_state.filtered_results is None
    )

    fee_range = st.slider(
        "Tuition Fee Range ($):",
        min_value=0,
        max_value=100000,
        value=(0, 100000),
        step=5000,
        disabled=st.session_state.filtered_results is None
    )

    min_duration = 1.0
    max_duration = 5.0

    duration_range = st.slider(
        "Duration (years):",
        min_value=min_duration,
        max_value=max_duration,
        value=(min_duration, max_duration),
        step=0.5,
        disabled=st.session_state.filtered_results is None
    )

    # We'll populate these filters after the first search
    degree_levels = ["All Levels"]
    if st.session_state.filtered_results is not None and not st.session_state.filtered_results.empty:
        unique_levels = get_unique_values(st.session_state.filtered_results, 'course_level')
        degree_levels += unique_levels

    st.session_state.filter_degree_levels = st.multiselect(
        "Degree Level:",
        options=degree_levels,
        default=["All Levels"],
        disabled=st.session_state.filtered_results is None
    )

    # Recent searches section
    st.markdown("<h2 class='subheader'>🕒 Recent Searches</h2>", unsafe_allow_html=True)

    if st.session_state.recent_searches:
        for i, recent_query in enumerate(st.session_state.recent_searches[-5:]):
            if st.button(f"{recent_query}", key=f"recent_{i}"):
                st.session_state.last_query = recent_query
                st.rerun()
    else:
        st.info("Your recent searches will appear here.")

    # Favorites
    st.markdown("<h2 class='subheader'>⭐ Favorites</h2>", unsafe_allow_html=True)

    if st.session_state.favorites:
        for i, fav in enumerate(st.session_state.favorites):
            cols = st.columns([4, 1])
            cols[0].write(f"{fav['course']} at {fav['university']}")
            if cols[1].button("🗑", key=f"del_fav_{i}"):
                st.session_state.favorites.pop(i)
                st.rerun()
    else:
        st.info("Star your favorite courses to save them here.")

# Main search interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "🔍 Search for courses, universities, or subjects:",
        value=st.session_state.last_query,
        placeholder="e.g., Computer Science at University of Melbourne",
        help="Enter keywords related to courses, universities, or study fields"
    )

with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

    if st.button("🔄 Clear", use_container_width=True):
        query = ""
        st.session_state.filtered_results = None
        st.session_state.last_query = ""
        st.rerun()

# Execute search when button is pressed
if search_button and query.strip():
    # Update recent searches
    if query not in st.session_state.recent_searches:
        st.session_state.recent_searches.append(query)
        # Keep only the last 10 searches
        st.session_state.recent_searches = st.session_state.recent_searches[-10:]

    st.session_state.last_query = query

    # Display a spinner while searching
    with st.spinner("🔍 Searching for courses..."):
        # Perform search based on selected method
        if search_method == "Semantic":
            results = ir_system.search(query, top_n=top_n, method="semantic")
        elif search_method == "Fuzzy":
            results = ir_system.search(query, top_n=top_n, method="fuzzy")
        else:
            results = ir_system.search(query, top_n=top_n)

        if results is not None and not results.empty:
            results = results.drop_duplicates(subset=['institution_name'], keep='first')

        # Store results for filtering
        st.session_state.filtered_results = results

        # Update filter options based on results
        if results is not None and not results.empty:
            # Update locations filter with options from search results
            locations = ["All Locations"] + get_unique_values(results, 'location_name')
            st.session_state.filter_locations = st.sidebar.multiselect(
                "Locations:",
                options=locations,
                default=["All Locations"],
                key="locations_update"
            )

            # Calculate min/max fee values from results
            if 'tuition_fee' in results.columns:
                min_fee = int(results['tuition_fee'].min()) if not results['tuition_fee'].empty else 0
                max_fee = int(results['tuition_fee'].max()) if not results['tuition_fee'].empty else 100000
                fee_range = st.sidebar.slider(
                    "Tuition Fee Range ($):",
                    min_value=min_fee,
                    max_value=max_fee,
                    value=(min_fee, max_fee),
                    step=5000,
                    key="fee_update"
                )

            # Calculate min/max duration values from results
            if 'duration_years' in results.columns:
                min_duration = float(results['duration_years'].min()) if not results['duration_years'].empty else 0.0
                max_duration = float(results['duration_years'].max()) if not results['duration_years'].empty else 8.0
                
                # Add a small buffer if min and max are the same
                if min_duration == max_duration:
                    min_duration = max(0.0, min_duration - 0.5)  # Subtract 0.5, but don't go below 0
                    max_duration = max_duration + 0.5  # Add 0.5 to max
                
                duration_range = st.sidebar.slider(
                    "Duration (years):",
                    min_value=min_duration,
                    max_value=max_duration,
                    value=(min_duration, max_duration),
                    step=0.5,
                    key="duration_update"
                )

# Apply filters to results
if st.session_state.filtered_results is not None and not st.session_state.filtered_results.empty:
    filtered_df = st.session_state.filtered_results.copy()

    # Apply location filter
    if st.session_state.filter_locations and "All Locations" not in st.session_state.filter_locations:
        filtered_df = filtered_df[filtered_df['location_name'].isin(st.session_state.filter_locations)]

    # Apply fee filter
    if 'tuition_fee' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['tuition_fee'] >= fee_range[0]) &
            (filtered_df['tuition_fee'] <= fee_range[1])
            ]

    # Apply duration filter
    if 'duration_years' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['duration_years'] >= duration_range[0]) &
            (filtered_df['duration_years'] <= duration_range[1])
            ]

    # Apply Degree Level filter
    if 'course_level' in filtered_df.columns and st.session_state.filter_degree_levels and "All Levels" not in st.session_state.filter_degree_levels:
        filtered_df = filtered_df[filtered_df['course_level'].isin(st.session_state.filter_degree_levels)]

    # Display results
    if not filtered_df.empty:
        st.success(f"✅ Found {len(filtered_df)} relevant courses for '{query}'!")

        # Results display options
        st.markdown("<h2 class='subheader'>🎯 Search Results</h2>", unsafe_allow_html=True)

        sort_col, sort_order = st.columns([2, 2])
        with sort_col:
            sort_by = st.selectbox(
                "Sort by:",
                options=["Relevance", "Tuition Fee (Low to High)", "Tuition Fee (High to Low)",
                         "Duration (Short to Long)", "Duration (Long to Short)"]
            )

        # Apply sorting
        if sort_by == "Tuition Fee (Low to High)" and 'tuition_fee' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('tuition_fee')
        elif sort_by == "Tuition Fee (High to Low)" and 'tuition_fee' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('tuition_fee', ascending=False)
        elif sort_by == "Duration (Short to Long)" and 'duration_years' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('duration_years')
        elif sort_by == "Duration (Long to Short)" and 'duration_years' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('duration_years', ascending=False)
        else:
            # Default sort by relevance (similarity_score)
            if 'similarity_score' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('similarity_score', ascending=False)

        # Display tabs for different view options
        tab1, tab2 = st.tabs(["🎴 Card View", "📋 Table View"])

        # Card view
        with tab1:
            for idx, (_, row) in enumerate(filtered_df.iterrows(), start=1):
                with st.container():
                    st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)

                    cols = st.columns([4, 1])
                    with cols[0]:
                        university = row.get('institution_name', 'N/A')
                        course = row.get('course_name', 'N/A')

                        st.markdown(f"<h3 class='course-name'>{course}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<h4 class='university-name'>{university}</h4>", unsafe_allow_html=True)

                    with cols[1]:
                        is_favorite = any(f['course'] == course and f['university'] == university
                                          for f in st.session_state.favorites)

                        if st.button("⭐" if not is_favorite else "★", key=f"fav_{idx}"):
                            if not is_favorite:
                                st.session_state.favorites.append({
                                    'course': course,
                                    'university': university
                                })
                            else:
                                st.session_state.favorites = [f for f in st.session_state.favorites
                                                              if not (
                                            f['course'] == course and f['university'] == university)]
                            st.rerun()

                    # Key metrics
                    metric_cols = st.columns(4)

                    with metric_cols[0]:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"📍 Location**<br>{row.get('location_name', 'N/A')}", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with metric_cols[1]:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"🕒 Duration**<br>{row.get('duration_years', 'N/A')} years",
                                    unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with metric_cols[2]:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"💰 Tuition Fee**<br>${int(row.get('tuition_fee', 0)):,}", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with metric_cols[3]:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"🔍 Relevance**<br>{round(row.get('similarity_score', 0) * 100, 1)}%",
                                    unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Expandable details
                    with st.expander("See Details"):
                        detail_cols = st.columns(2)

                        with detail_cols[0]:
                            st.markdown(f"📍 **Full Address:** {row.get('full_address', 'N/A')}")
                            st.markdown(f"📚 **Field(s) of Education:** {row.get('fields_of_education', 'N/A')}")
                            st.markdown(f"🧑‍💼 **Work Component:** {row.get('work_component_details', 'N/A')}")

                        # Add website link in the second column
                        with detail_cols[1]:
                            # More robust URL handling - add this before the website link code
                            university_url = row.get('Website', '')

                            # Cleanup URL if needed
                            if university_url:
                                # Remove any extra whitespace
                                university_url = university_url.strip()
                                
                                # Ensure URL has proper prefix
                                if not university_url.startswith(('http://', 'https://')):
                                    university_url = 'https://' + university_url
                                
                                # Display link with proper text
                                institution_name = row.get('institution_name', 'University')
                                st.markdown(f"* [Visit {institution_name} Website]({university_url})")
                            else:
                                # Fallback to search
                                search_query = row.get('institution_name', 'university').replace(' ', '+')
                                st.markdown(f"* [Search for University Website](https://www.google.com/search?q={search_query}+official+website)")

                    st.markdown("</div>", unsafe_allow_html=True)

        # Table view
        with tab2:
            # Prepare a clean version of the DataFrame for display
            display_cols = ['institution_name', 'course_name', 'location_name',
                            'duration_years', 'tuition_fee', 'similarity_score']

            display_cols = [col for col in display_cols if col in filtered_df.columns]

            if display_cols:
                # Rename columns for display
                display_df = filtered_df[display_cols].copy()
                column_map = {
                    'institution_name': 'University',
                    'course_name': 'Course',
                    'location_name': 'Location',
                    'duration_years': 'Duration (years)',
                    'tuition_fee': 'Tuition Fee ($)',
                    'similarity_score': 'Relevance'
                }

                display_df = display_df.rename(columns={k: v for k, v in column_map.items() if k in display_df.columns})

                # Format the relevance score as percentage
                if 'Relevance' in display_df.columns:
                    display_df['Relevance'] = display_df['Relevance'].apply(lambda x: f"{round(x * 100, 1)}%")

                # Display the table
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # --- Export Buttons ---
                st.markdown("### 📤 Export Results")

                # Create columns for export buttons
                col_export_csv, col_export_excel = st.columns([1, 1])

                # CSV export
                csv_data = display_df.to_csv(index=False).encode('utf-8')
                with col_export_csv:
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name="search_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # Excel export
                import io
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    display_df.to_excel(writer, index=False)
                excel_buffer.seek(0)
                excel_data = excel_buffer.getvalue()

                with col_export_excel:
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name="search_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

    else:
        st.warning("⚠ No results match your filters. Try adjusting your filter criteria.")
elif st.session_state.last_query and not st.session_state.filtered_results:
    st.warning(f"⚠ No results found for '{st.session_state.last_query}'. Try a different search query.")
else:
    st.info("👆 Enter a search query above to find courses at Australian universities.")

    # Show sample queries to help users get started
    st.markdown("<h3 class='subheader'>📝 Sample Queries</h3>", unsafe_allow_html=True)

    sample_queries = {
        "Computer Science": ["Australian Catholic University Limited - Bachelor of Computer Science",
                             "Monash University (Monash) - Doctor of Philosophy (IITB-Monash)"],

        "Business": ["University of Queensland - Bachelor of Business Management",
                     "Monash University - Bachelor of Commerce"],

        "Music": ["Excelsia University College - Bachelor of Music",
                  "Macquarie University (Macquarie) - Bachelor of Music"],

        "Psychology": ["Macquarie University (Macquarie) - Bachelor of Psychology",
                       "University of New England - Bachelor of Psychology with Honours"],

        "Journalism": ["Monash University (Monash) - Master of Journalism",
                       "Macquarie University (Macquarie) - Master of Media Studies"],

        "Law": ["Macquarie University (Macquarie) - Master of International Law, Governance and Public Policy",
                "Top Education Group Ltd - Bachelor of Law"]
    }

    sample_cols = st.columns(len(sample_queries))

    for i, (col, query) in enumerate(zip(sample_cols, sample_queries)):
        if col.button(query, key=f"sample_{i}"):
            st.session_state.last_query = query
            st.rerun()

# Footer with information
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
    🇦🇺 Australian Universities Course Search | Data sourced from Australian higher education institutions
    </div>
    """,
    unsafe_allow_html=True
)


def render_filters(results_df):
    ...


# call it only once, outside of the button logic
if st.session_state.filtered_results is not None:
    render_filters(st.session_state.filtered_results)