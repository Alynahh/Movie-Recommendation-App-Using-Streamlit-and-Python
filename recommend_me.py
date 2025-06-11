import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 💫 Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# 🧼 Text cleaning function
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = str(text).lower()
    words = word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# 📦 Load dataset (with poster URLs)
data = pd.read_csv('cleaned_netflix_movies_with_posters.csv')

# 🎯 Fallback for review column
if 'Rotten Tomatoes Reviews' in data.columns:
    review_column = 'Rotten Tomatoes Reviews'
elif 'IMDB Reviews' in data.columns:
    review_column = 'IMDB Reviews'
else:
    review_column = 'Tags'

# 🧃 Simulate sentiment labels (optional: replace with real ones if available)
data['Sentiment'] = np.random.choice(['positive', 'negative'], size=len(data))

# 🔍 TF-IDF vectorization for tags
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['Tags'])

# 🧠 Main recommendation functions
def recommend_movies(user_input, top_n=5):
    cleaned_input = clean_text(user_input)
    user_vector = tfidf.transform([cleaned_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return data.iloc[top_indices][['Title', 'Genre', 'Tags', 'Poster']]

def rule_based_recommendation(user_genre, top_n=5):
    user_genre = user_genre.lower()
    filtered = data[data['Genre'].str.contains(user_genre, na=False)]
    if filtered.empty:
        return data.sample(top_n)[['Title', 'Genre', 'Tags', 'Poster']]
    return filtered.sample(min(top_n, len(filtered)))[['Title', 'Genre', 'Tags', 'Poster']]

# ❤️ Sentiment-based model training
def train_sentiment_model():
    X = data[review_column].fillna('')
    y = data['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectors = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vectors, y_train)
    return model, vectorizer

sentiment_model, review_vectorizer = train_sentiment_model()

# 🌈 Recommend positive-vibe movies
def recommend_positive_movies(user_query, top_n=5):
    top_movies = recommend_movies(user_query, top_n=20)
    reviews = top_movies['Tags']
    vectors = review_vectorizer.transform(reviews)
    sentiments = sentiment_model.predict(vectors)
    positive_indices = [i for i, s in enumerate(sentiments) if s == 'positive']
    if not positive_indices:
        return top_movies.head(top_n)
    return top_movies.iloc[positive_indices].head(top_n)
