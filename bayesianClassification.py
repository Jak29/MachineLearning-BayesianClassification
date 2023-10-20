import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# splitting and counting the reviews
def task1():
    # Reading the file
    movie_reviews = pd.read_excel("movie_reviews.xlsx")
        
    # Training data, containing all reviews of the training set
    training_data = movie_reviews.loc[movie_reviews["Split"] == "train", "Review"]
    # Training labels, containing all associated sentiment labels for the training data
    training_labels = movie_reviews.loc[movie_reviews["Split"] == "train", "Sentiment"]
    # Test data, containing all reviews of the test set
    test_data = movie_reviews.loc[movie_reviews["Split"] == "test", "Review"]
    # Test labels, containing all associated sentiment labels for the test data
    test_labels = movie_reviews.loc[movie_reviews["Split"] == "test", "Sentiment"]
    
    # The number of positive reviews in the training set
    training_positive_count = (training_labels == "positive").sum()
    # The number of negative reviews in the training set
    training_negative_count = (training_labels == "negative").sum()
    # The number of positive reviews in the evaluation set
    test_positive_count = (test_labels == "positive").sum()
    # The number of negative reviews in the evaluation set
    test_negative_count = (test_labels == "negative").sum()
    
    print("positive reviews in training set: ", training_positive_count)
    print("negative reviews in training set: ", training_negative_count )
    print("positive reviews in test set: ", test_positive_count)
    print("negative reviews in test set: ", test_negative_count)

    return training_data, training_labels, test_data, test_labels, training_positive_count, training_negative_count

    
# extract relevant features
def task2(training_data, min_len, min_occur):
    # Removing the apostrophe ' first so I can keep the words together example they're -> ["theyre"] instead of ["they", "re"]
    training_data = training_data.str.replace("'", "")
    # Replacing non alphanumeric characters with spaces
    training_data = training_data.replace('[^a-zA-Z0-9]', ' ', regex=True)
    # Convert the data to lowercase
    training_data = training_data.str.lower()
    # Split the reviews into a list of words
    training_data = training_data.str.split()
    
    # List that will contain every word
    all = []
    
    # For every review extend its list of words to the "all" list of words
    # (Must be a better way to do this very slow)
    for review in training_data:
        all.extend(review)
    
    # Count the occurences of each word
    occurrences = pd.value_counts(np.array(all))
    
    # List that will be returned with relevant features
    occurrences_cleanup = []
    
    # For every word in occurences and the amount of times it appears
    for word, appears in occurrences.items():
        # If the word appears more or equal to the minimum amount of occurrences
        # and the word is longer or equal to the minimum length
        # add it to the filtered series
        if appears >= min_occur and len(word) >= min_len:
            occurrences_cleanup.append(word)
            
    # Return the list of features
    return occurrences_cleanup


# count feature frequencies
def task3(words, review):
    # Get the reviews with a positive sentiment
    positive = review.loc[review["Sentiment"] == "positive"]
    # Get the reviews with a negative sentiment
    negative = review.loc[review["Sentiment"] == "negative"]

    # Get rid of the sentiment column
    positive = positive.drop("Sentiment", axis=1).squeeze()
    # Get rid of the sentiment column
    negative = negative.drop("Sentiment", axis=1).squeeze()
    
    # Removing the apostrophe ' first so I can keep the words together example they're -> ["theyre"] instead of ["they", "re"]
    positive = positive.str.replace("'", "")
    # Replacing non alphanumeric characters with spaces
    positive = positive.replace('[^a-zA-Z0-9]', ' ', regex=True)
    # Convert the data to lowercase
    positive = positive.str.lower()
    
    negative = negative.str.replace("'", "")
    negative = negative.replace('[^a-zA-Z0-9]', ' ', regex=True)
    negative = negative.str.lower()
    
    # Dictionary containing the number of reviews every word occurred in
    positive_dict = {}
    negative_dict = {}

    # For a word in thee list of words
    for word in words:
        # Set the amount of occurences for the word to 0
        positive_dict[word] = 0
        negative_dict[word] = 0
        
        # For a review in the positive reviews
        for review in positive:
            # If the word appears in the review
            if word in review:
                # Add 1 to its count
                positive_dict[word] = positive_dict.get(word) + 1
             
        # For a review in the negative reviews
        for review in negative:
            # If the word appears in the review
            if word in review:
                # Add 1 to its count
                negative_dict[word] = negative_dict.get(word) + 1
        
    # Return the dictionaries 
    return (positive_dict, negative_dict)


# calculate feature likelihoods and priors
def task4(positive_dict, negative_dict, training_positive_count, training_negative_count):
    
    # Dictionary of the words and likelihoods 
    positive_prob_dict = {}
    negative_prob_dict = {}
    
    # Likelihood that a word is in a positive review 
    # For every word in the positive dictionary containing the count of reviews every word occurred in
    for word in positive_dict:
        # Laplace smoothing
        # Add 1 to the count of positive reviews containg a word
        # Add 1 to the count of the total positive reviews
        # Get the likelihood that word is in a positive review 
        # Add the word and its likelihoods to a dict 
        positive_prob_dict[word] = (positive_dict[word]+1) / (training_positive_count+1)
        
    # Likelihood that a word is in a negative review 
    # For every word in the positive dictionary containing the count of reviews every word occurred in
    for word in negative_dict:#
        # Laplace smoothing with a smoothing factor 𝛼 = 1
        # Add 1 to the count of negative reviews containg a word
        # Add 1 to the count of the total negative reviews
        # Get the likelihood that word is in a negative review 
        # Add the word and its likelihoods to a dict 
        negative_prob_dict[word] = (negative_dict[word]+1) / (training_negative_count+1)
        
    # Priors
    # Divide the total count of reviews by the count of the positive or negative reviews
    positive_prob_review = training_positive_count / (training_negative_count + training_positive_count)
    negative_prob_review= training_negative_count / (training_negative_count + training_positive_count)
    
    # positive_prob_dict the dict that contains the likelihood of each word appearing in a positive review
    # negative_prob_dict the dict that contains the likelihood of each word appearing in a negative review
    # positive_prob_review the positive prior
    # negative_prob_review the negative prior
    return positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review


# maximum likelihood classification
def task5(positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, new_review):
    
    # Dictionary of the words and likelihoods but log
    positive_prob_log_dict = {}
    negative_prob_log_dict = {}
    
    # For a word in the positive likelihood dictionary
    for word in positive_prob_dict:
        # For a word in the positive likelihood dictionary
        positive_prob_log_dict[word] = math.log(positive_prob_dict[word])
        
    # For a word in the negative likelihood dictionary
    for word in negative_prob_dict:
        # For a word in the negative likelihood dictionary
        negative_prob_log_dict[word] = math.log(negative_prob_dict[word])
        
    # String containing a review
    # Removing the apostrophe ' first so I can keep the words together example they're -> ["theyre"] instead of ["they", "re"]  
    new_review = str(new_review).replace("'", "")
    # Replacing non alphanumeric characters with spaces
    # Convert to lowercase
    # Split the review into a list of words
    new_review = new_review.replace('[^a-zA-Z0-9]', ' ').lower().split()
    
    
    # Variables for the likelihood
    log_likelihood_positive = 0
    log_likelihood_negative = 0
    
    # For a word in the review
    for word in new_review:  
        # If the word is in the positive dictionary
        if word in positive_prob_log_dict:         
            # Using the words positive likelihood I get thhe reviews positive likelihood  (conditional independence)
            log_likelihood_positive = log_likelihood_positive * positive_prob_log_dict[word]
            
        # If the word is in the negative dictionary
        if word in negative_prob_log_dict:  
            # Using the words negative likelihood I get thhe reviews negative likelihood  (conditional independence)
            log_likelihood_negative = log_likelihood_negative * negative_prob_log_dict[word]
        
    # Include the priors
    log_likelihood_positive = log_likelihood_negative * math.log(positive_prob_review)
    log_likelihood_negative = log_likelihood_positive * math.log(negative_prob_review)
    
    # Convert the log probability back to normal
    positive = math.exp(log_likelihood_positive)
    negative = math.exp(log_likelihood_negative)
    
    # See which probability is greater positive or negative
    predicted_sentiment = 'positive' if positive > negative else 'negative'
    
    # Return sentiment that is more likely
    return predicted_sentiment


# Used instead of task5 because it can take a dataset instead of a single string
def predict(positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, reviews):
    
    positive_prob_log_dict = {}
    negative_prob_log_dict = {}
    predicted_sentiment = []
    
    for word in positive_prob_dict:
        positive_prob_log_dict[word] = math.log(positive_prob_dict[word])
        
    for word in negative_prob_dict:
        negative_prob_log_dict[word] = math.log(negative_prob_dict[word])
    
    
    # Same as task5 but just going through each review and converting into a string first
    for review in reviews:
            
        words = str(review).replace("'", "")
        words = words.replace('[^a-zA-Z0-9]', ' ').lower().split()
        
        log_likelihood_positive = 0
        log_likelihood_negative = 0
        
        for word in words:  
            if word in positive_prob_log_dict:            
                log_likelihood_positive = log_likelihood_positive + positive_prob_log_dict[word]
            
            
        for word in words:
            if word in negative_prob_log_dict:  
                log_likelihood_negative = log_likelihood_negative + negative_prob_log_dict[word]
            
        log_likelihood_positive += math.log(positive_prob_review)
        log_likelihood_negative += math.log(negative_prob_review)
        
        positive = math.exp(log_likelihood_positive)
        negative = math.exp(log_likelihood_negative)
        
        sentiment = 'positive' if positive > negative else 'negative'
        predicted_sentiment.append(sentiment)
        
    
    return predicted_sentiment
    

# evaluation of results
def task6(training_data, training_labels, positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, training_positive_count, training_negative_count):
    # Reseting the index dropping the old index and reshaping to a 2d array with 1 column (the review)
    training_data = training_data.reset_index(drop=True)
    training_data = training_data.to_numpy().reshape(-1, 1)
    # Resetting the index dropping the old index and reshaping to a 2d array with 1 column (the sentiment)
    training_labels = training_labels.reset_index(drop=True)
    training_labels = training_labels.to_numpy().reshape(-1, 1)
    
    # Creating the KFold classifier
    kf = model_selection.KFold(n_splits=500,shuffle=True)
    mean_accuracy = []
    
    for k in range(1,5):
        # Lists that will reset for every fold
        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []
        
        for train_index, test_index in kf.split(training_data, training_labels):
            # List of the predictions for the split
            # Will reset every split
            # Predict is task5 but reads in datasets instead of a string
            predicted_labels = predict(positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, training_data[test_index])
    
            # Calculating confusion matrix
            C = metrics.confusion_matrix(training_labels[test_index], predicted_labels)
            true_positives.append(C[0,0])
            true_negatives.append(C[1,1])
            false_positives.append(C[1,0])
            false_negatives.append(C[0,1])
                
        # K-Fold cross-validation results
        sum_true_positives = sum(true_positives)
        sum_true_negatives = sum(true_negatives)
        sum_false_positives = sum(false_positives)
        sum_false_negatives = sum(false_negatives)
        total = sum_false_negatives + sum_false_positives + sum_true_negatives + sum_true_positives
        
        print("k = ", k)
        print("True positives: ", sum_true_positives, " ", round(sum_true_positives / total * 100, 2), "%")
        print("True negatives: ", sum_true_negatives, " ", round(sum_true_negatives / total * 100, 2), "%")
        print("False positives: ", sum_false_positives, " ", round(sum_false_positives / total * 100, 2), "%")
        print("False negatives: ", sum_false_negatives, " ", round(sum_false_negatives / total * 100, 2), "%")
        print("Accuracy: ", (sum(true_positives)+sum(true_negatives))/(training_positive_count+training_negative_count))        
        print("----------------------")
        print()
        
        mean_accuracy.append((sum(true_positives)+sum(true_negatives))/(training_positive_count+training_negative_count))
    
    print("Mean Accuracy: ", sum(mean_accuracy) / len(mean_accuracy))
    
    
# evaluation of results on test data
def task6part2(test_data, test_labels, positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, training_positive_count, training_negative_count):
    print("Test Data")
    print("----------------------")
    # Reseting the index dropping the old index and reshaping to a 2d array with 1 column (the review)
    test_data = test_data.reset_index(drop=True)
    test_data = test_data.to_numpy().reshape(-1, 1)
    # Resetting the index dropping the old index and reshaping to a 2d array with 1 column (the sentiment)
    test_labels = test_labels.reset_index(drop=True)
    test_labels = test_labels.to_numpy().reshape(-1, 1)
    
    # Creating the KFold classifier
    kf = model_selection.KFold(n_splits=500,shuffle=True)
    mean_accuracy = []
    
    for k in range(1,5):
        # Lists that will reset for every fold
        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []
        
        for train_index, test_index in kf.split(test_data, test_labels):
            # List of the predictions for the split
            # Will reset every split
            # Predict is task5 but reads in datasets instead of a string
            predicted_labels = predict(positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, test_data[test_index])
    
            # Calculating confusion matrix
            C = metrics.confusion_matrix(test_labels[test_index], predicted_labels)
            true_positives.append(C[0,0])
            true_negatives.append(C[1,1])
            false_positives.append(C[1,0])
            false_negatives.append(C[0,1])
                    
        # K-Fold cross-validation results
        sum_true_positives = sum(true_positives)
        sum_true_negatives = sum(true_negatives)
        sum_false_positives = sum(false_positives)
        sum_false_negatives = sum(false_negatives)
        total = sum_false_negatives + sum_false_positives + sum_true_negatives + sum_true_positives
        
        print("k = ", k)
        print("True positives: ", sum_true_positives, " ", round(sum_true_positives / total * 100, 2), "%")
        print("True negatives: ", sum_true_negatives, " ", round(sum_true_negatives / total * 100, 2), "%")
        print("False positives: ", sum_false_positives, " ", round(sum_false_positives / total * 100, 2), "%")
        print("False negatives: ", sum_false_negatives, " ", round(sum_false_negatives / total * 100, 2), "%")
        print("Accuracy: ", (sum(true_positives)+sum(true_negatives))/(training_positive_count+training_negative_count))       
        print("Confusion Matrix: ", C)
        print("----------------------")
        print()
        
    mean_accuracy.append((sum(true_positives)+sum(true_negatives))/(training_positive_count+training_negative_count))
    
    print("Mean Accuracy: ", sum(mean_accuracy) / len(mean_accuracy))
    
    
def main():
    # Task 1
    training_data, training_labels, test_data, test_labels, training_positive_count, training_negative_count = task1()
    
    # Task 2
    LENGTH = 4
    OCCURRENCE = 1000
    words = task2(training_data, LENGTH, OCCURRENCE)
    print("Length: ", LENGTH)
    print("Occurrence: ", OCCURRENCE)
    print("----------------------")
    
    # Task 3
    review = pd.concat([training_data, training_labels], axis=1)
    positive_dict, negative_dict = task3(words, review)
    
    # Task 4
    positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review = task4(positive_dict, negative_dict, training_positive_count, training_negative_count)
    
    # Task 5
    new_review = "After seeing PURELY BELTER I came onto this site to review it , but not only that I also had to check out the resume of the screenwriter / director Mark Herman . As soon as his name appeared on the opening credits I knew that I had seen his name before somewhere and after checking I found out he wrote and directed the film version of LITTLE VOICE one of the most underrated feelgood British movies of the 1990s   PURELY BELTER is an entirely different kettle of fish . It's a grim stereotypical view of Geordie life and a very unfunny one at that . Everyone is either a wife beater , a single mother , a shoplifter , a drunk or a junkie . Since many scenes are set in a school the PE teacher is a sadistic bully and that's the closest the film ever gets to reality . Oh and everyone is very foul mouthed which adds to the grim unlikable atmosphere  I didn't like PURELY BELTER much while I watched and now that I know who Mark Herman is I like it even less . With LITTLE VOICE Herman proved you can make an amusing uplifting comedy featuring northern souls but I had to ask where his undoubted talent went in this movie ?"
    predicted_sentiment = task5(positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, new_review)
    
    # Task 6
    
    task6(training_data, training_labels, positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, training_positive_count, training_negative_count)
    # The Final evaluation including configuration matrix and uses test data
    task6part2(test_data, test_labels, positive_prob_dict, negative_prob_dict, positive_prob_review, negative_prob_review, training_positive_count, training_negative_count)

    
main()