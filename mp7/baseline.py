"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    # count the number of times each word is tagged with each tag
    word_tag_counts = {}
    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag_counts:
                word_tag_counts[word] = {}
            if tag not in word_tag_counts[word]:
                word_tag_counts[word][tag] = 0
            word_tag_counts[word][tag] += 1

    # for each word, find the tag it is most commonly tagged with
    most_common_tag = {}
    for word in word_tag_counts:
        most_common_tag[word] = max(word_tag_counts[word], key=word_tag_counts[word].get)

    # count the number of times each tag is used
    tag_counts = {}
    for sentence in train:
        for word, tag in sentence:
            if tag not in tag_counts:
                tag_counts[tag] = 0
            tag_counts[tag] += 1

    # find the most common tag
    most_common_tag_all = max(tag_counts, key=tag_counts.get)

    # predict the most common tag for each word in the test set
    predicts = []
    for sentence in test:
        predicts.append([(word, most_common_tag.get(word, most_common_tag_all)) for word in sentence])
    
    return predicts
