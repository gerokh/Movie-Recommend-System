import requests
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# =============================
# CONFIG & SETUP
# =============================
load_dotenv()

st.set_page_config(
    page_title="Cinéma Luxe",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API KEYS & PATHS
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_PATH = "https://image.tmdb.org/t/p/w780"

# Local Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

if not TMDB_API_KEY:
    st.error("⚠️ TMDB_API_KEY is missing! Please check your .env file.")
    st.stop()

# =============================
# DATA LOADING (CACHED)
# =============================
@st.cache_resource
def load_data():
    """Loads the pickle files and initializes the recommender data."""
    try:
        with open(DF_PATH, "rb") as f:
            df = pickle.load(f)
        
        with open(INDICES_PATH, "rb") as f:
            indices = pickle.load(f)
            
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
            
        # Normalize indices map keys
        title_to_idx = {}
        # Handle if indices is dict or series
        iterable = indices.items() if hasattr(indices, "items") else zip(indices.index, indices.values)
        for k, v in iterable:
            title_to_idx[str(k).strip().lower()] = int(v)
            
        return df, title_to_idx, tfidf_matrix
    except FileNotFoundError as e:
        st.error(f"❌ Critical file missing: {e}. Please ensure .pkl files are in the directory.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}. You might need to update 'numpy' or check pickle compatibility.")
        return None, None, None

df, title_to_idx, tfidf_matrix = load_data()


# =============================
# LOGIC & ALGORITHMS
# =============================
def get_recommendations(title, top_n=10):
    """Generates content-based recommendations using TF-IDF."""
    if df is None or tfidf_matrix is None:
        return []
    
    # Normalize title
    norm_title = title.strip().lower()
    
    if norm_title not in title_to_idx:
        return []

    idx = title_to_idx[norm_title]
    
    # Calculate similarity scores
    # tfidf_matrix is sparse, so we use dot product
    sig = tfidf_matrix[idx]
    # Cosine similarity (assuming normalized vectors)
    sim_scores = (tfidf_matrix @ sig.T).toarray().flatten()
    
    # Sort
    sim_indices = sim_scores.argsort()[::-1][1:top_n+1] # Exclude self (index 0 usually, but argsort is low->high, so reverse)
    
    # Get titles
    movie_indices = [i for i in sim_indices]
    return df['title'].iloc[movie_indices].tolist()

# =============================
# TMDB API HELPERS
# =============================
@st.cache_data(ttl=3600)
def fetch_tmdb(endpoint, params=None):
    if params is None: 
        params = {}
    params["api_key"] = TMDB_API_KEY
    params["language"] = "en-US"
    
    try:
        response = requests.get(f"{TMDB_BASE_URL}{endpoint}", params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_movie_details(tmdb_id):
    return fetch_tmdb(f"/movie/{tmdb_id}")

@st.cache_data(ttl=3600)
def search_tmdb(query):
    return fetch_tmdb("/search/movie", {"query": query, "include_adult": "false"})

@st.cache_data(ttl=3600)
def get_poster_url(title):
    # Helper to find a poster by title if we only have the title (from TF-IDF)
    data = search_tmdb(title)
    if data and data.get("results"):
        img_path = data["results"][0].get("poster_path")
        tmdb_id = data["results"][0].get("id")
        return (f"{TMDB_IMG_PATH}{img_path}" if img_path else None), tmdb_id
    return None, None

# =============================
# CUSTOM CSS (GLASSMORPHISM)
# =============================
def local_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: #ffffff;
        }
        
        .stApp {
            background-image: linear-gradient(to right top, #0f0c29, #302b63, #24243e);
            background-size: cover;
            background-attachment: fixed;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 24px;
            margin-bottom: 24px;
        }
        
        .poster-title {
            margin-top: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            color: #e0e0e0;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
local_css()


if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

def navigate_to(view_name, movie_id=None):
    st.session_state.view = view_name
    if movie_id:
        st.session_state.selected_movie_id = movie_id
    st.rerun()


def render_movie_grid(movies, title="Movies"):
    if not movies:
        st.info("No movies found.")
        return

    st.markdown(f"### {title}")
    cols = st.columns(6)
    
    for idx, movie in enumerate(movies):
        with cols[idx % 6]:
            
            poster = movie.get("poster_url") or (f"{TMDB_IMG_PATH}{movie['poster_path']}" if movie.get("poster_path") else None)
            if not poster:
                poster = "https://via.placeholder.com/500x750?text=No+Image"
                
            mtit = movie.get("title", "Untitled")
            mid = movie.get("id") or movie.get("tmdb_id")

            st.image(poster, use_column_width=True)
            if st.button("View", key=f"btn_{mid}_{idx}"):
                navigate_to("details", mid)
            st.markdown(f"<div class='poster-title'>{mtit}</div>", unsafe_allow_html=True)
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)


def view_home():
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🎬 CINÉMA LUXE</h1>", unsafe_allow_html=True)
    
    
    query = st.text_input("Search movie...", placeholder="Type title then press enter")
    if query:
        
        data = search_tmdb(query)
        if data and data.get("results"):
            render_movie_grid(data["results"], "Search Results")
        else:
            st.warning("No results found.")
    
    st.divider()
    
    
    tabs = st.tabs(["🔥 Trending", "✨ Popular", "⭐ Top Rated"])
    
    with tabs[0]:
        data = fetch_tmdb("/trending/movie/day")
        if data: render_movie_grid(data.get("results", [])[:12], "")

    with tabs[1]:
        data = fetch_tmdb("/movie/popular")
        if data: render_movie_grid(data.get("results", [])[:12], "")

    with tabs[2]:
        data = fetch_tmdb("/movie/top_rated")
        if data: render_movie_grid(data.get("results", [])[:12], "")

def view_details():
    mid = st.session_state.selected_movie_id
    if not mid:
        navigate_to("home")
        return

    details = get_movie_details(mid)
    if not details:
        st.error("Failed to load details.")
        if st.button("Back"): navigate_to("home")
        return

    # Backdrop setup
    if details.get("backdrop_path"):
        bd_url = f"{TMDB_IMG_PATH}{details['backdrop_path']}"
        st.markdown(f"""
        <style>
        .stApp:before {{
            content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: url('{bd_url}') no-repeat center center fixed;
            background-size: cover; opacity: 0.15; z-index: -1; filter: blur(8px);
        }}
        </style>
        """, unsafe_allow_html=True)

    if st.button("← Back to Home"):
        navigate_to("home")

    st.markdown(f"# {details['title']}")
    
    c1, c2 = st.columns([1, 2], gap="large")
    
    with c1:
        poster = f"{TMDB_IMG_PATH}{details['poster_path']}" if details.get("poster_path") else None
        if poster: st.image(poster, use_column_width=True)
        
        st.write(f"**Date:** {details.get('release_date')}")
        st.write(f"**Rating:** {details.get('vote_average')}/10")

    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Overview")
        st.write(details.get("overview", "No overview."))
        st.markdown("</div>", unsafe_allow_html=True)
        
        
        st.markdown("### More Like This")
        
       
        rec_movies = []
        titles = get_recommendations(details['title']) # Local logic
        
        if titles:
            st.success(f"Found {len(titles)} content-based recommendations!")
            
            for t in titles:
                p_url, t_id = get_poster_url(t)
                if t_id:
                    rec_movies.append({"title": t, "poster_url": p_url, "tmdb_id": t_id})
        else:
           
            st.info("No content match found in local DB. Showing TMDB recommendations.")
            tmdb_recs = fetch_tmdb(f"/movie/{mid}/recommendations")
            if tmdb_recs:
                rec_movies = tmdb_recs.get("results", [])

        if rec_movies:
            render_movie_grid(rec_movies[:12], "")
        else:
            st.warning("No recommendations available.")


if st.session_state.view == "home":
    view_home()
elif st.session_state.view == "details":
    view_details()
