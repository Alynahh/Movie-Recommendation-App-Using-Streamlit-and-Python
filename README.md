# Movie-Recommendation-App-Using-Streamlit-and-Python
This is a mood-based movie recommendation web app built using Python and Streamlit. Users can enter their mood, favorite genre, and keywords to receive personalized movie suggestions powered by natural language processing and machine learning.

##  Features

*  Keyword-based content recommendations using TF-IDF + Cosine Similarity
*  Genre-based rule filtering
*  Sentiment-aware movie filtering (recommends only positively reviewed films)
*  Clean, modern UI using Streamlit and optional custom CSS
*  Automatic preprocessing of the movie dataset (text cleaning, stemming, stopword removal)

---

##  Folder Structure

```
Movie-Recommendation-App/
├── recommend_me.py                 # Main Streamlit app
├── movie_logic.py                  # ML logic: recommend_movies, etc.
├── preprocess.py                   # Data cleaning script
├── cleaned_netflix_movies.csv      # Preprocessed dataset (generated)
├── netflix-rotten-tomatoes.csv     # Original dataset (input)
├── style.css                       # Optional styling for Streamlit
├── requirements.txt                # Python dependencies
└── README.md                       # You're here!
```

---

##  How to Run

### 1. Clone the Repo

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Clean the Dataset

Run the preprocessing script to generate a cleaned version of the movie data:

```bash
python preprocess.py
```

### 4. Launch the Web App

```bash
streamlit run recommend_me.py
```

A browser window will open at [http://localhost:8501](http://localhost:8501)

---

##  How it Works

The recommendation system includes 3 approaches:

| Mode                  | Description                                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Smart (ML)            | Uses TF-IDF + cosine similarity on cleaned movie "Tags"                                                                       |
| Genre-Based           | Filters movies using string matching on genre field                                                                           |
| Feel-Good (Sentiment) | Filters recommendations to only show movies with positive review sentiment (trained using logistic regression on review data) |

---

##  Built With

* Python 3.13
* Streamlit
* scikit-learn
* NLTK
* pandas, numpy

---

##  Contributors

* Aleena – Frontend & Integration 
* Raima – Sentiment Analysis Model
* Nooram – Data Preprocessing
* Fahaam – Recommendation Logic (TF-IDF & Genre)

---

##  License

This project is for academic purposes. You are free to fork and modify it for personal or educational use.

