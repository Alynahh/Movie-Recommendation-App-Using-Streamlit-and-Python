# preprocess.py
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    text = str(text).lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def preprocess_dataset(file_path):
    data = pd.read_csv(file_path)
    data.dropna(subset=['Title', 'Genre', 'Tags', 'Series or Movie'], inplace=True)
    data.drop_duplicates(subset='Title', keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)

    data['Tags'] = data['Tags'].apply(clean_text)
    data['Genre'] = data['Genre'].apply(lambda x: clean_text(x))

    if 'Rotten Tomatoes Reviews' in data.columns:
        data['Rotten Tomatoes Reviews'] = data['Rotten Tomatoes Reviews'].apply(lambda x: clean_text(x) if pd.notnull(x) else '')
    if 'IMDB Reviews' in data.columns:
        data['IMDB Reviews'] = data['IMDB Reviews'].apply(lambda x: clean_text(x) if pd.notnull(x) else '')

    data.to_csv('cleaned_netflix_movies.csv', index=False)
    return data
if __name__ == "__main__":
    preprocess_dataset("netflix-rotten-tomatoes-dataset.csv")  # Replace with your original CSV filename
