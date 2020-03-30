from flask import Flask, request, jsonify, render_template
from app import app
from app.demographic import demographic
from app.recommendations import movie_recommendations
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trending',methods=['GET'])
def predict():
    data1 = os.path.join(BASE_DIR, "app", "data", "tmdb_5000_credits.csv")
    data2 = os.path.join(BASE_DIR, "app", "data", "tmdb_5000_movies.csv")
    titles, scores = demographic(data1,data2)

    data = []
    for i in range(len(titles)):
        data.append({'title': titles[i], 'score': scores[i]})
    # print(data)
    return render_template('demographic.html', data=data)

@app.route('/recommendations', methods=['GET'])
def render_recommendations():
    return render_template('recommendations.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    title = request.form.get('movie_title')
    data = os.path.join(BASE_DIR, "app", "data", "netflix_movies.csv")
    recommendations = movie_recommendations(Title=title, data_path=data)
    return render_template('recommendations.html', movies=recommendations)