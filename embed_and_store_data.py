from sentence_transformers import SentenceTransformer
import pandas as pd
from elasticsearch import Elasticsearch, helpers
movies = pd.read_csv("cleaned_imdb_top_1000.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
def generate_embeddings(text):
    return model.encode(text, convert_to_tensor=False).tolist()
movies['Overview_embeddings'] = movies['Overview'].apply(generate_embeddings)
es = Elasticsearch("http://localhost:9200", basic_auth=('elastic', 'kM_NA4l_euNlOPe9JgUl'))
def index_documents(index_name, documents):
    actions = [
        {
            "_index": index_name,
            "_id": doc["Series_Title"],
            "_source": {
                "Series_Title": doc["Series_Title"],
                "Overview": doc["Overview"],
                "Genre": doc["Genre"],
                "Director": doc["Director"],
                "IMDB_Rating": doc["IMDB_Rating"],
                "embedding": doc["Overview_embeddings"]
            }
        }
        for doc in documents.to_dict(orient="records")
    ]

    helpers.bulk(es, actions)
    print(f"Indexed {len(documents)} documents into {index_name}")
index_name = "movies"
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)
index_documents(index_name, movies)
