"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict, Counter
from math import log
import enum


class WordType(enum.Enum):
    NUMERIC = 1
    VERY_SHORT = 2
    SHORT_S = 3
    SHORT_OTHER = 4
    SHORT_ED = 7
    LONG_S = 5
    LONG_OTHER = 6
    LONG_ED = 8


def word_type(word) -> WordType:
    if word[0].isdigit() and word[-1].isdigit():
        return WordType.NUMERIC
    if len(word) <= 3:
        return WordType.VERY_SHORT
    if 4 <= len(word) <= 9:
        if word[-1] == 's':
            return WordType.SHORT_S
        elif word[-2:] == 'ed':
            return WordType.SHORT_ED
        else:
            return WordType.SHORT_OTHER
    if len(word) >= 10:
        if word[-1] == 's':
            return WordType.LONG_S
        elif word[-2:] == 'ed':
            return WordType.LONG_ED
        else:
            return WordType.LONG_OTHER


# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5  # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0.)  # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0.))  # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0.))  # {tag0: {tag1: # }}
    emit_count = defaultdict(lambda: defaultdict(lambda: 0))  # {tag: {word: # }}
    trans_count = defaultdict(lambda: defaultdict(lambda: 0.))  # {tag0: {tag1: # }}

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    alpha_e = 0.00001  # smoothing parameter
    alpha_t = 0.00001

    # init_prob
    init_prob['START'] = 1

    # emit_prob and trans_prob
    for sentence in sentences:
        prev_tag = 'START'
        for word, tag in sentence:
            emit_count[tag][word] += 1
            trans_count[prev_tag][tag] += 1
            prev_tag = tag
        trans_count[prev_tag]['END'] += 1

    # extract words that appear only once in training data
    hapax_count = defaultdict(lambda: defaultdict(lambda: 1))
    for tag in emit_count:
        for word in emit_count[tag]:
            if emit_count[tag][word] == 1:
                hapax_count[tag][word_type(word)] += 1
    hapax_count_total = sum(sum(hapax_count[tag].values()) for tag in hapax_count)
    tag_scale = defaultdict(lambda: defaultdict(lambda: 0.))
    for tag in emit_count:  # all tags
        for type in WordType:  # all types
            tag_scale[tag][type] = hapax_count[tag][type] / hapax_count_total

    # normalize emit_prob
    for tag in emit_count:
        n_t = sum(emit_count[tag].values())  # total number of words in training data for tag T
        v_t = len(emit_count[tag])  # number of unique words seen in training data for tag T
        for word in emit_count[tag]:
            tag_word_alpha_e = alpha_e * tag_scale[tag][word_type(word)]
            emit_prob[tag][word] = (emit_count[tag][word] + tag_word_alpha_e) / (n_t + tag_word_alpha_e * (v_t + 1))
        for type in WordType:
            tag_alpha_e = alpha_e * tag_scale[tag][type]
            emit_prob[tag][f"{type}"] = tag_alpha_e / (n_t + tag_alpha_e * (v_t + 1))

    # normalize trans_prob
    for prev_tag in emit_prob:
        for tag in emit_prob:
            if trans_count[prev_tag][tag] == 0:
                trans_count[prev_tag][tag] = alpha_t
        total_transitions = sum(trans_count[prev_tag].values())
        for tag in trans_count[prev_tag]:
            trans_prob[prev_tag][tag] = trans_count[prev_tag][tag] / total_transitions

    return init_prob, emit_prob, trans_prob


def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {}  # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {}  # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    if i == 0:
        for tag in emit_prob:
            if word in emit_prob[tag]:
                log_prob[tag] = log(emit_prob[tag][word])
            else:
                log_prob[tag] = log(emit_prob[tag][f"{word_type(word)}"])
            predict_tag_seq[tag] = [tag]
        return log_prob, predict_tag_seq

    # general case (i > 0)
    for tag in emit_prob:
        max_prob = float('-inf')
        best_prev_tag = None
        for prev_tag in prev_prob:
            trans_p = trans_prob[prev_tag][tag]
            if word in emit_prob[tag]:
                emit_p = emit_prob[tag][word]
            else:
                emit_p = emit_prob[tag][f"{word_type(word)}"]
            prob = prev_prob[prev_tag] + math.log(trans_p) + math.log(emit_p)
            if prob > max_prob:
                max_prob = prob
                best_prev_tag = prev_tag
        log_prob[tag] = max_prob
        predict_tag_seq[tag] = prev_predict_tag_seq[best_prev_tag] + [tag]

    return log_prob, predict_tag_seq


def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)

    predicts = []

    for sen in range(len(test)):
        sentence = test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq,
                                                            emit_prob, trans_prob)

        # TODO:(III)
        # according to the storage of probabilities and sequences, get the final prediction.
        best_tag = max(log_prob, key=log_prob.get)
        predicts.append([(sentence[i], predict_tag_seq[best_tag][i]) for i in range(length)])

    return predicts
