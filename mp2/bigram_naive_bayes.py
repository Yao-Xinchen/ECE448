# bigram_naive_bayes.py
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
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.005, bigram_laplace=0.005, bigram_lambda=0.5, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    uni_pos_log_prob, uni_neg_log_prob = train_unigram(unigram_laplace, train_labels, train_set)
    bi_pos_log_prob, bi_neg_log_prob = train_bigram(bigram_laplace, train_labels, train_set)

    yhats = dev(dev_set, uni_neg_log_prob, uni_pos_log_prob, bi_neg_log_prob, bi_pos_log_prob, bigram_lambda, pos_prior, silently)

    return yhats


def train_unigram(laplace, train_labels, train_set):
    # count words in positive and negative reviews
    pos_count = Counter()
    neg_count = Counter()
    for i in range(len(train_set)):
        if train_labels[i] == 1:
            pos_count.update(train_set[i])
        else:
            neg_count.update(train_set[i])

    # calculate probabilities
    pos_total, neg_total = sum(pos_count.values()), sum(neg_count.values())
    neg_words, pos_words = set(neg_count.keys()), set(pos_count.keys())
    pos_prob = {word: (pos_count[word] + laplace) / (pos_total + laplace * (len(pos_words) + 1)) for word in pos_words}
    neg_prob = {word: (neg_count[word] + laplace) / (neg_total + laplace * (len(neg_words) + 1)) for word in neg_words}
    pos_prob["UNK"] = laplace / (pos_total + laplace * (len(pos_words) + 1))
    neg_prob["UNK"] = laplace / (neg_total + laplace * (len(neg_words) + 1))

    # convert to log probabilities
    pos_log_prob = {word: math.log(prob) for word, prob in pos_prob.items()}
    neg_log_prob = {word: math.log(prob) for word, prob in neg_prob.items()}

    return pos_log_prob, neg_log_prob


def train_bigram(laplace, train_labels, train_set):
    # count bigrams in positive and negative reviews
    pos_count = Counter()
    neg_count = Counter()
    for i in range(len(train_set)):
        bigram_set = list(zip(train_set[i], train_set[i][1:]))
        if train_labels[i] == 1:
            pos_count.update(bigram_set)
        else:
            neg_count.update(bigram_set)

    # calculate probabilities
    pos_total, neg_total = sum(pos_count.values()), sum(neg_count.values())
    neg_bigrams, pos_bigrams = set(neg_count.keys()), set(pos_count.keys())
    pos_prob = {bigram: (pos_count[bigram] + laplace) / (pos_total + laplace * (len(pos_bigrams) + 1)) for bigram in pos_bigrams}
    neg_prob = {bigram: (neg_count[bigram] + laplace) / (neg_total + laplace * (len(neg_bigrams) + 1)) for bigram in neg_bigrams}
    pos_prob["UNK"] = laplace / (pos_total + laplace * (len(pos_bigrams) + 1))
    neg_prob["UNK"] = laplace / (neg_total + laplace * (len(neg_bigrams) + 1))

    # convert to log probabilities
    pos_log_prob = {bigram: math.log(prob) for bigram, prob in pos_prob.items()}
    neg_log_prob = {bigram: math.log(prob) for bigram, prob in neg_prob.items()}

    return pos_log_prob, neg_log_prob


def dev(dev_set, uni_neg_log_prob, uni_pos_log_prob, bi_neg_log_prob, bi_pos_log_prob, bigram_lambda, pos_prior, silently):
    yhats = []

    for doc in tqdm(dev_set, disable=silently):
        # unigram score
        uni_pos_score = math.log(pos_prior)
        uni_neg_score = math.log(1 - pos_prior)
        for word in doc:
            # pos
            if word in uni_pos_log_prob:
                uni_pos_score += uni_pos_log_prob[word]
            else:
                uni_pos_score += uni_pos_log_prob["UNK"]
            # neg
            if word in uni_neg_log_prob:
                uni_neg_score += uni_neg_log_prob[word]
            else:
                uni_neg_score += uni_neg_log_prob["UNK"]
        # bigram score
        bi_pos_score = math.log(pos_prior)
        bi_neg_score = math.log(1 - pos_prior)
        for bigram in zip(doc, doc[1:]):
            # pos
            if bigram in bi_pos_log_prob:
                bi_pos_score += bi_pos_log_prob[bigram]
            else:
                bi_pos_score += bi_pos_log_prob["UNK"]
            # neg
            if bigram in bi_neg_log_prob:
                bi_neg_score += bi_neg_log_prob[bigram]
            else:
                bi_neg_score += bi_neg_log_prob["UNK"]
        # combine scores
        pos_score = (1 - bigram_lambda) * uni_pos_score + bigram_lambda * bi_pos_score
        neg_score = (1 - bigram_lambda) * uni_neg_score + bigram_lambda * bi_neg_score
        yhats.append(1 if pos_score > neg_score else 0)

    return yhats