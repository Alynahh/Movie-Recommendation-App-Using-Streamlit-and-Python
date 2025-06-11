import streamlit as st
from recommend_me import recommend_movies, rule_based_recommendation, recommend_positive_movies

st.set_page_config(page_title="🍿 Aleena's Movie Recommender ✨", page_icon="🎬", layout="centered")

# Optional cute background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff3f3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title section
st.title("🍿 Movie Recommendation Magic ✨")
st.subheader("Find the perfect movie for your mood!")

# User input form
with st.form("recommendation_form"):
    mood = st.text_input("💖 What's your mood today?")
    genre = st.text_input("🎬 Favorite movie genre?")
    keyword = st.text_input("🔎 Any keyword you want in your movie?")
    method = st.radio("How should I recommend?", ["Smart (TF-IDF)", "Genre-Based", "Feel-Good (Sentiment)"])
    submitted = st.form_submit_button("✨ Recommend Me Movies ✨")

# Process form input
if submitted:
    st.success(f"Searching movies for mood: **{mood}**, genre: **{genre}**, keyword: **{keyword}** 🧠")

    query = f"{mood} {genre} {keyword}"

    # Recommendation based on selected method
    if method == "Smart (TF-IDF)":
        results = recommend_movies(query)
    elif method == "Genre-Based":
        results = rule_based_recommendation(genre)
    else:
        results = recommend_positive_movies(query)

    st.write("💖 Here are your top movie matches:")
    for i, row in results.iterrows():
        st.write(f"🎬 **{row['Title']}** — _{row['Genre']}_")


