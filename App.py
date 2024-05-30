import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3

# Load the dataset
dataset = pd.read_csv('IMDb Top TV Series.csv')
dataset.head(5)
dataset.info()

# Handle missing values
dataset.fillna('', inplace=True)

# Extract relevant columns
dataset = dataset[['Title', 'Year', 'Parental Rating', 'Rating', 'Number of Votes', 'Description']]

# Normalize numerical features
dataset['Rating'] = dataset['Rating'].astype(float)


# Function to convert 'Number of Votes' to an integer
def convert_votes(votes):
    if 'M' in votes:
        return int(float(votes.replace('M', '')) * 1_000_000)
    elif 'K' in votes:
        return int(float(votes.replace('K', '')) * 1_000)
    else:
        return int(votes)


# Apply the function to the 'Number of Votes' column
dataset['Number of Votes'] = dataset['Number of Votes'].apply(convert_votes)

# Remove leading numbers and any formatting from the 'Title' column
dataset['Title'] = dataset['Title'].str.replace(r'^\d+\.\s+', '', regex=True)

# Encode categorical features
dataset['Parental Rating'] = dataset['Parental Rating'].astype('category').cat.codes

# Converting descriptions into numerical vectors
# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the description column
tfidf_matrix = tfidf.fit_transform(dataset['Description'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Function to get the index of a TV series from its title
def get_index_from_title(title):
    return dataset[dataset['Title'] == title].index.values[0]


# Function to get the title of a TV series from its index
def get_title_from_index(index):
    return dataset.iloc[index]['Title']


# Function to recommend TV series
def recommend_tv_series(title, num_recommendations=5):
    # Get the index of the TV series that matches the title
    idx = get_index_from_title(title)

    # Get the pairwise similarity scores for all TV series
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the TV series based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar TV series
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the TV series titles
    tv_series_indices = [i[0] for i in sim_scores]
    recommendations = [get_title_from_index(i) for i in tv_series_indices]

    return recommendations


print("\nRecommendations based on 'Breaking Bad':", recommend_tv_series("Breaking Bad", 5))
print("Recommendations based on 'House':",recommend_tv_series("House", 5))
print("Recommendations based on 'Monk':",recommend_tv_series("Monk", 5))
