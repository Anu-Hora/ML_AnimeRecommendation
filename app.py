import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Anime Recommender System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Anime Content-Based Recommender System")
st.write("Recommend similar anime using Content-Based Filtering (Genre + Type).")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    anime = pd.read_csv("anime.csv")

    anime['genre'] = anime['genre'].fillna("Unknown")
    anime['type'] = anime['type'].fillna("Unknown")
    anime['rating'] = anime['rating'].fillna(anime['rating'].median())

    anime['episodes'] = anime['episodes'].replace("Unknown", np.nan)
    anime['episodes'] = pd.to_numeric(anime['episodes'], errors='coerce')
    anime['episodes'] = anime['episodes'].fillna(anime['episodes'].median())

    anime = anime.drop_duplicates()

    return anime


anime = load_data()

# -----------------------------
# Feature Engineering
# -----------------------------
anime["content"] = (
    anime["genre"].astype(str)
    + " "
    + anime["type"].astype(str)
)

# -----------------------------
# TF-IDF
# -----------------------------
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(anime["content"])

# -----------------------------
# Cosine Similarity
# -----------------------------
cosine_sim = cosine_similarity(tfidf_matrix)

indices = pd.Series(
    anime.index,
    index=anime["name"]
).drop_duplicates()


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(title, n=10):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:n+1]

    anime_indices = [i[0] for i in sim_scores]

    result = anime.iloc[anime_indices][
        ["name",
         "genre",
         "type",
         "episodes",
         "rating",
         "members"]
    ].copy()

    result["Similarity"] = [
        round(score[1],3)
        for score in sim_scores
    ]

    return result


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Search Anime")

selected_anime = st.sidebar.selectbox(
    "Choose an Anime",
    sorted(anime["name"].unique())
)

num = st.sidebar.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

# -----------------------------
# Selected Anime Details
# -----------------------------
row = anime[anime["name"] == selected_anime].iloc[0]

st.subheader("Selected Anime")

c1, c2 = st.columns(2)

with c1:
    st.write("**Name:**", row["name"])
    st.write("**Genre:**", row["genre"])
    st.write("**Type:**", row["type"])

with c2:
    st.write("**Episodes:**", row["episodes"])
    st.write("**Rating:**", row["rating"])
    st.write("**Members:**", f"{int(row['members']):,}")

# -----------------------------
# Recommendation Button
# -----------------------------
if st.button("Recommend Similar Anime"):

    rec = recommend(selected_anime, num)

    st.success(f"Top {num} Recommendations")

    st.dataframe(
        rec,
        use_container_width=True
    )

# -----------------------------
# Dataset Preview
# -----------------------------
with st.expander("View Dataset"):
    st.dataframe(anime.head(20))