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
    
    training_positive_count = (training_labels == "positive").sum()
    training_negative_count = (training_labels == "negative").sum()
    test_positive_count = (test_labels == "positive").sum()
    test_negative_count = (test_labels == "negative").sum()
    
    # print("positive reviews in training set: ", training_positive_count)
    # print("negative reviews in training set: ", training_negative_count )
    # print("positive reviews in test set: ", test_positive_count)
    # print("negative reviews in test set: ", test_negative_count)

    
    return training_data, training_labels, test_data, test_labels, training_positive_count, training_negative_count

    
def task2(training_data, min_len, min_occur):
    training_data = training_data.str.replace("'", "")
    training_data = training_data.replace('[^a-zA-Z0-9]', ' ', regex=True)
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
            occurences_cleanup[word] = occur#
    
    return occurences_cleanup.index[:]

    
def task3(words, review):
    positive = review.loc[review["Sentiment"] == "positive"]
    negative = review.loc[review["Sentiment"] == "negative"]

    positive = positive.drop("Sentiment", axis=1).squeeze()
    negative = negative.drop("Sentiment", axis=1).squeeze()
    
    positive = positive.str.replace("'", "")
    positive = positive.replace('[^a-zA-Z0-9]', ' ', regex=True)
    positive = positive.str.lower()
    negative = negative.str.replace("'", "")
    negative = negative.replace('[^a-zA-Z0-9]', ' ', regex=True)
    negative = negative.str.lower()
    
    positive_dict = {}
    negative_dict = {}
    
    for word in words:
        positive_dict[word] = 0
        negative_dict[word] = 0
        for review in positive:
            if word in review:
                positive_dict[word] = positive_dict.get(word) + 1
        for review in negative:
            if word in review:
                negative_dict[word] = negative_dict.get(word) + 1
        
    return (positive_dict, negative_dict)


def task4(positive_dict, negative_dict, training_positive_count, training_negative_count):
    positive_prob_dict = {}
    negative_prob_dict = {}
    
    for word in positive_dict:
        positive_prob_dict[word] = (positive_dict[word]+1) / (training_positive_count+1)
        
    for word in negative_dict:
        negative_prob_dict[word] = (negative_dict[word]+1) / (training_negative_count+1)
        
    positive_prob_review = training_positive_count / (training_negative_count + training_positive_count)
    negative_prob_review= training_negative_count / (training_negative_count + training_positive_count)
    
    print(positive_prob_review)
    print(negative_prob_review)
    
    return positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review


def task5(positive_prob_review, negative_prob_review):
    
    
    
    
def main():
    # Task 1
    training_data, training_labels, test_data, test_labels, training_positive_count, training_negative_count = task1()
    
    # Task 2
    words = task2(training_data, 4, 1000)
    
    # Task 3
    review = pd.concat([training_data, training_labels], axis=1)
    positive_dict, negative_dict = task3(words, review)    
    
    # Task 4
    positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review = task4(positive_dict, negative_dict, training_positive_count, training_negative_count)
    
    # Task 5
    task5(positive_prob_review, negative_prob_review)
    
main()