import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
movies = pd.read_csv("imdb_top_1000.csv")
text_columns = ['Series_Title', 'Genre', 'Released_Year', 'Certificate', 'Runtime', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross']
def clean_text(text):
    if not isinstance(text, str):
        return 'Unknown'
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.lower()
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text
for col in text_columns:
    movies[col] = movies[col].apply(clean_text)
numeric_columns = ['Meta_score', 'No_of_Votes']
for col in numeric_columns:
    movies[col] = pd.to_numeric(movies[col], errors='coerce').fillna(0)
movies.drop_duplicates(subset=['Series_Title'], inplace=True)
movies.to_csv("cleaned_imdb_top_1000.csv", index=False)
