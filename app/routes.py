from flask import Flask, request, jsonify, render_template
from app import app
from app.demographc import demographic
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    data1 = os.path.join(BASE_DIR, "app", "data", "tmdb_5000_credits.csv")
    data2 = os.path.join(BASE_DIR, "app", "data", "tmdb_5000_movies.csv")
    titles, scores = demographic(data1,data2)
    return render_template('titles.html', titles=titles, scores=scores)