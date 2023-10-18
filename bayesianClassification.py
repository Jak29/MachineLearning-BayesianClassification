import numpy as np
import pandas as pd
import math
from sklearn import metrics


def task1():
    movie_reviews = pd.read_excel("movie_reviews.xlsx")
    
    training_data = movie_reviews.loc[movie_reviews["Split"] == "train", "Review"]
    training_labels = movie_reviews.loc[movie_reviews["Split"] == "train", "Sentiment"]
    test_data = movie_reviews.loc[movie_reviews["Split"] == "test", "Review"]
    test_labels = movie_reviews.loc[movie_reviews["Split"] == "test", "Sentiment"]
    
    print("positive reviews in training set: ", (training_labels == "positive").sum())
    print("negative reviews in training set: ",(training_labels == "negative").sum())
    print("positive reviews in test set: ",(test_labels == "positive").sum())
    print("negative reviews in test set: ",(test_labels == "negative").sum())

    
    return training_data, training_labels, test_data, test_labels
    
def task2():
    


def main():
    training_data, training_labels, test_data, test_labels = task1()
    task2()
    
main()