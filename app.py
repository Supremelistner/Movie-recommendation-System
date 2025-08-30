from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import emoji
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import wikipedia
import wikipediaapi

app = Flask(__name__)

def tokenizer(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = emoji.demojize(text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'#', '', text)  
    text = re.sub(r'RT[\s]+', '', text)  
    text = re.sub(r'https?:\/\/\S+', '', text) 
    words = text.lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

base_dir = os.path.dirname(__file__)
tfi = pickle.load(open(os.path.join(base_dir, "tfid.pkl"), "rb"))
matrix_tf = pickle.load(open(os.path.join(base_dir, "matrix_tf.pkl"), "rb"))

df = pd.read_csv("C:\\Users\\MANISH\\OneDrive\\Desktop\\Dataset\\MovieSummaries\\movie_project.csv", encoding="utf-8")

import wikipedia

def get_movie_image_and_link(movie_name):
    try:
        search_results = wikipedia.search(movie_name)
        if not search_results:
            return "https://via.placeholder.com/200x300?text=No+Image", "#"

        page = wikipedia.page(search_results[0])
        url = page.url

        images = page.images
        img_url = next((img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png')) and 'poster' in img.lower()), None)

        return img_url or "https://via.placeholder.com/200x300?text=No+Image", url

    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0])
            url = page.url
            return "https://via.placeholder.com/200x300?text=Image+Unavailable", url
        except:
            return "https://via.placeholder.com/200x300?text=Image+Unavailable", "#"

    except wikipedia.exceptions.PageError:
        return "https://via.placeholder.com/200x300?text=No+Image", "#"

    except Exception as e:
        print(f"Unexpected error fetching wiki data for {movie_name}: {e}")
        return "https://via.placeholder.com/200x300?text=No+Image", "#"


    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0])
            url = page.url
            return "https://via.placeholder.com/200x300?text=Image+Unavailable", url
        except:
            return "https://via.placeholder.com/200x300?text=Image+Unavailable", "#"

    except wikipedia.exceptions.PageError:
        return "https://via.placeholder.com/200x300?text=No+Image", "#"

    except Exception as e:
        print(f"Unexpected error fetching wiki data for {movie_name}: {e}")
        return "https://via.placeholder.com/200x300?text=No+Image", "#"


def get_top_5_movies(prompt):
    prompt_vector = tfi.transform([prompt])
    scores = cosine_similarity(prompt_vector, matrix_tf).flatten()
    top_indices = scores.argsort()[-5:][::-1]
    movies = df.iloc[top_indices][['Movie name', 'Movie release date', 'PlotSummary']]

    for i, movie in movies.iterrows():
        img_url, url = get_movie_image_and_link(movie['Movie name'])
        movies.at[i, 'Image URL'] = img_url or "https://via.placeholder.com/200x300?text=No+Image"
        movies.at[i, 'Wikipedia Link'] = url

    return movies

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        prompt = request.form["prompt"]
        results = get_top_5_movies(prompt).to_dict(orient="records")
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
