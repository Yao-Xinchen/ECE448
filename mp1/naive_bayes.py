# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    pos_log_prob, neg_log_prob = train(laplace, train_labels, train_set)

    yhats = dev(dev_set, neg_log_prob, pos_log_prob, pos_prior, silently)

    return yhats


"""
Returns the log probabilities of each word in the positive and negative classes.
"""
def train(laplace, train_labels, train_set):
    # count words in positive and negative reviews
    pos_count = Counter()
    neg_count = Counter()
    for i in range(len(train_set)):
        if train_labels[i] == 1:
            pos_count.update(train_set[i])
        else:
            neg_count.update(train_set[i])

    # calculate probabilities
    pos_total = sum(pos_count.values())
    neg_total = sum(neg_count.values())
    neg_words, pos_words = set(neg_count.keys()), set(pos_count.keys())
    vocab = neg_words.union(pos_words)
    vocab_size = len(vocab)
    pos_prob = {word: (pos_count[word] + laplace) / (pos_total + laplace * vocab_size) for word in vocab}
    neg_prob = {word: (neg_count[word] + laplace) / (neg_total + laplace * vocab_size) for word in vocab}

    # convert to log probabilities
    pos_log_prob = {word: math.log(prob) for word, prob in pos_prob.items()}
    neg_log_prob = {word: math.log(prob) for word, prob in neg_prob.items()}

    return pos_log_prob, neg_log_prob


"""
Calculates the probability of each review being positive according to the model.
Returns the predicted labels for the dev set.
"""
def dev(dev_set, neg_log_prob, pos_log_prob, pos_prior, silently):
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        # for each review, calculate the score for positive and negative classes
        pos_score = math.log(pos_prior)
        neg_score = math.log(1 - pos_prior)  # neg_prior = 1 - pos_prior
        for word in doc:
            if word in pos_log_prob:
                pos_score += pos_log_prob[word]
            if word in neg_log_prob:
                neg_score += neg_log_prob[word]
        yhats.append(1 if pos_score > neg_score else 0)
    return yhats
