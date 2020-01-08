import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Reading CSV File
df= pd.read_csv("movie_preprocessed_dataset.csv")
###print (df.columns)

## Select the features from the CSV file for better results that you can sort by. 
features=['keywords','cast','genres','director']

##Create a column in DF which combines all selected features
for feature in features:
    df[feature]=df[feature].fillna('')
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

df["combined_features"]=df.apply(combine_features,axis=1)
print(df["combined_features"].head())


##Step 4: Create count matrix from this new combined column
cv=CountVectorizer()
count_matrix=cv.fit_transform(df["combined_features"])
##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim=cosine_similarity(count_matrix)


##Choose the Movie Name from the CSV file want to get results from PS: I have used Men In Black Movie ;)

movie_user_likes = "Men in Black 3"

##Get index of this movie from its title
movie_index=get_index_from_title(movie_user_likes)
similar_movies=list(enumerate(cosine_sim[movie_index]))

##Get a list of similar movies in descending order of similarity score
sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

##Print titles of first 20 alike movies you can change the values according to need
i=0
for movie in sorted_similar_movies:
    print (get_title_from_index(movie[0]))
    i=i+1
    if i>20:
        break
   
