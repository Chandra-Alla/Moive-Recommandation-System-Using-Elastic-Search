import streamlit as st
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import pandas as pd
st.set_page_config(page_title="Movie Search Engine", layout="wide")
model = SentenceTransformer('all-MiniLM-L6-v2')
es = Elasticsearch("http://localhost:9200", basic_auth=('elastic', 'JAxYKwUIqbAm=EM54ZC8'))


def search_similar_movies(query, top_n=10):
    query_embedding = model.encode([query])[0].tolist()

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding}
            }
        }
    }

    try:
        response = es.search(
            index="movies",
            body={
                "query": script_query,
                "size": top_n,
                "_source": ["Series_Title", "Director", "IMDB_Rating", "Genre"]
            }
        )

        movies_found = [
            {
                "movie_name": hit['_source'].get('Series_Title', 'Unknown'),
                "director": hit['_source'].get('Director', 'Unknown'),
                "imdb_rating": hit['_source'].get('IMDB_Rating', 'Unknown'),
                "genre": hit['_source'].get('Genre', 'Unknown')
            }
            for hit in response['hits']['hits']
        ]

        return pd.DataFrame(movies_found)

    except Exception as e:
        st.error(f"Failed to query Elasticsearch: {str(e)}")
        return pd.DataFrame()


# Streamlit UI
st.title('🎥 Movie Search Engine')
user_query = st.text_input("Enter a movie description:", "")

if user_query:
    with st.spinner('Searching for similar movies...'):
        similar_movies = search_similar_movies(user_query, top_n=5)
        if not similar_movies.empty:
            st.write('Similar movies found:')
            st.dataframe(similar_movies)
        else:
            st.write('No similar movies found. Please try another description.')
