import numpy as np
import pandas as pd
import math
from sklearn import metrics
import re
from collections import Counter

def task1():
    movie_reviews = pd.read_excel("movie_reviews.xlsx")
    
    training_data = movie_reviews.loc[movie_reviews["Split"] == "train", "Review"]
    training_labels = movie_reviews.loc[movie_reviews["Split"] == "train", "Sentiment"]
    test_data = movie_reviews.loc[movie_reviews["Split"] == "test", "Review"]
    test_labels = movie_reviews.loc[movie_reviews["Split"] == "test", "Sentiment"]
    
    # print("positive reviews in training set: ", (training_labels == "positive").sum())
    # print("negative reviews in training set: ",(training_labels == "negative").sum())
    # print("positive reviews in test set: ",(test_labels == "positive").sum())
    # print("negative reviews in test set: ",(test_labels == "negative").sum())

    
    return training_data, training_labels, test_data, test_labels

    
def task2(training_data, min_len, min_occur):
    training_data = training_data.replace('[^a-zA-Z0-9 ]', ' ', regex=True)
    training_data = training_data.str.lower()
    training_data = training_data.str.split()
    
    all = []
    
    for review in training_data:
        all.extend(review)
        
    occurrences = pd.value_counts(np.array(all))
    occurences_cleanup = pd.Series()
    
    # Quite slow so keep min_len and min_occur values high
    for word, occur in occurrences.items():
        if occur >= min_occur and len(word) >= min_len:
            occurences_cleanup[word] = occur
    
    return occurences_cleanup.index[:]

    
def task3(words, review):
    positive = review.loc[review["Sentiment"] == "positive"]
    negative = review.loc[review["Sentiment"] == "negative"]
    
    positive_dict = {}
    negative_dict = {}
    
    for word in words:
        positive_dict[word] = positive_dict.get(word, 0) + 1
        
    for word in words:
        negative_dict[word] = negative_dict.get(word, 0) + 1
    
     
    
    
def main():
    training_data, training_labels, test_data, test_labels = task1()
    words = task2(training_data, 15, 20)
    review = pd.concat([training_data, training_labels], axis=1)
    task3(words, review)    
main()