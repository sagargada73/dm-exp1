import pandas as pd
import numpy as np
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def movie_recommendations(Title, data_path):
    movies_df = pd.read_csv(data_path)
    new_movies_df = movies_df[['title', 'director', 'cast', 'listed_in', 'description']]

    # REMOVE NaN VALUES AND EMPTY STRINGS:
    new_movies_df.dropna(inplace=True)
    blanks = []
    cols = ['title', 'director', 'cast', 'listed_in', 'description']
    for i, col in new_movies_df.iterrows():
        if type(col) == str:
            if col.isspace():
                blanks.append(i)
    new_movies_df.drop(blanks, inplace=True)

    new_movies_df['key_words'] = ""

    for index, row in new_movies_df.iterrows():
        description = row['description']

        # instantiating Rake, by default it uses english stopwords from NLTK
        # and discards all puntuation characters as well

        r = Rake()
        
        # extracting the words by passing the text
        r.extract_keywords_from_text(description)

        # getting the dictionary whith key words as keys and their scores as values
        key_words_dict_scores = r.get_word_degrees()

        # assigning the key words to the new column for the corresponding movie
        row['key_words'] = list(key_words_dict_scores.keys())

    # dropping the Plot column
    new_movies_df.drop(columns = ['description'], inplace = True)

    # discarding the commas between the actors' full names and getting only the first three names
    new_movies_df['cast'] = new_movies_df['cast'].map(lambda x: x.split(',')[:3])

    # putting the genres in a list of words
    new_movies_df['listed_in'] = new_movies_df['listed_in'].map(lambda x: x.lower().split(','))

    new_movies_df['director'] = new_movies_df['director'].map(lambda x: x.split(' '))

    # merging together first and last name for each actor and director, so it's considered as one word 
    # and there is no mix up between people sharing a first name
    for index, row in new_movies_df.iterrows():
        row['cast'] = [x.lower().replace(' ','') for x in row['cast']]
        row['director'] = ''.join(row['director']).lower()

    new_movies_df.set_index('title', inplace = True)

    new_movies_df['bag_of_words'] = ''
    columns = new_movies_df.columns
    for index, row in new_movies_df.iterrows():
        words = ''
        for col in columns:
            if col != 'director':
                words = words + ' '.join(row[col])+ ' '
            else:
                words = words + row[col]+ ' '
        row['bag_of_words'] = words
        
    new_movies_df.drop(columns = [col for col in new_movies_df.columns if col!= 'bag_of_words' and col != 'type'], inplace = True)

    # instantiating and generating the count matrix
    movies_count = CountVectorizer()
    movies_count_matrix = movies_count.fit_transform(new_movies_df['bag_of_words'])

    # creating a Series for the movie titles so they are associated to an ordered numerical
    # list I will use later to match the indexes
    movies_indices = pd.Series(new_movies_df.index)
    
    # generating the cosine similarity matrix
    movies_cosine_sim = cosine_similarity(movies_count_matrix, movies_count_matrix)

    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = movies_indices[movies_indices == Title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(movies_cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(new_movies_df.index)[i])
        
    return recommended_movies
