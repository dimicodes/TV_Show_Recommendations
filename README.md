# TV_Show_Recommendations

This project demonstrates a simple content-based recommendation system for TV series using Python. The system recommends TV series based on the description of a given TV series using TF-IDF vectorization and cosine similarity.

## Dataset
The dataset used is IMDb Top TV Series.csv, which includes the following columns:

**Title**: The name of the TV series.
**Year**: The release year of the series.
**Parental Rating**: The content rating of the series.
**Rating**: The IMDb rating of the series.
**Number of Votes**: The total number of votes received by the series.
**Description**: A brief synopsis of the series.

## Project Overview

**Data Loading and Preprocessing:**
- Load the dataset.
- Handle missing values.
- Extract relevant columns.
- Normalize numerical features.
- Convert the Number of Votes column to integer.
- Remove leading numbers and any formatting from the Title column.
- Encode categorical features.

**TF-IDF Vectorization:**
- Convert the Description column into numerical vectors using TF-IDF vectorization.

**Cosine Similarity Calculation:**
- Compute the cosine similarity matrix for the TV series descriptions.

**Recommendation Function:**
- Define a function to get the index of a TV series from its title.
- Define a function to get the title of a TV series from its index.
- Define a function to recommend TV series based on the similarity scores.
