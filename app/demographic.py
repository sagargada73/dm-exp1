import pandas as pd
import numpy as np
import pickle

def demographic(data1,data2):
    df1=pd.read_csv(data1)
    df2=pd.read_csv(data2)
    # print(df1)
    df1.columns = ['id','title','cast','crew']
    df2= df2.merge(df1,on='id')

    C= df2['vote_average'].mean()
    m= df2['vote_count'].quantile(0.9)
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)
    
    
    q_movies = df2.copy().loc[df2['vote_count'] >= m]
    # weighted_rating(x,m,c)
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    q_movies = q_movies.sort_values('score', ascending=False)

    trained_data=q_movies[['original_title', 'vote_count', 'vote_average', 'score']].head(10)

    pop= q_movies.sort_values('score', ascending=False)
    return pop['original_title'].head(15).tolist() , pop['score'].head(15).tolist() 