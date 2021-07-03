"""
Dedicated to doctor professor Oren Kurland
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import math
import pandas as pd
stop_words = set(stopwords.words('english'))


def get_num_of_words(list_of_sentences):
    num = 0
    for sen in list_of_sentences:
        num += len(sen.split())
    return num


def calc_tf_score(list_of_sentences, num_of_words):
    tf_score = defaultdict(int)
    for sen in list_of_sentences:
        for word in sen:
            if word not in stop_words:
                tf_score[word] += 1
    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y / int(num_of_words)) for x, y in tf_score.items())
    return tf_score


def check_sent(word, sentences):
    final = [all([w in x for w in word]) for x in sentences]
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))


def calc_idf_score(list_of_sentences, num_of_sens):
    idf_score = defaultdict(int)
    for sen in list_of_sentences:
        for word in sen:
            if word not in stop_words:
                idf_score[word] = check_sent(word, list_of_sentences)
    # Dividing by total_word_length for each dictionary element
    idf_score.update((x, math.log(int(num_of_sens)/y)) for x, y in idf_score.items())
    return idf_score


def tf_idf_dict(list_of_sentences):
    """
    calc tfidf score to all words in a list of sentences
    :param list_of_sentences: the list of all sentences
    :return: tf_idf score dict
    """
    num_of_sen = len(list_of_sentences)
    num_of_words = get_num_of_words(list_of_sentences)
    tf_score = calc_tf_score(list_of_sentences, num_of_words)
    idf_score = calc_idf_score(list_of_sentences, num_of_sen)
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    return tf_idf_score


def get_key_words_for_sen(sen, tf_idf_score, lower_threshold, upper_threshold=10000000):
    """
    get keywords for a sentence
    :param sen: the sentence
    :param tf_idf_score: tf_idf_score dict like above
    :param lower_threshold: lower bound on tfidf score
    :param upper_threshold: upper bound on tfidf score
    :return: keywords for sentence sen
    """
    keywords = []
    for word in sen:
        if tf_idf_score[word] > lower_threshold and tf_idf_score[word] < upper_threshold:
            keywords.append(word)
    return keywords


def pipeline(list_of_sentences, lower_threshold, upper_threshold=10000000):
    """
    do everything above for a list of sentences
    :param list_of_sentences: the list of all sentences
    :param lower_threshold: lower bound on tfidf score
    :param upper_threshold: upper bound on tfidf score
    :return: nothing
    """
    tf_idf_score = tf_idf_dict(list_of_sentences)
    key_list = []
    for sentence in list_of_sentences:
        key_list.append(get_key_words_for_sen(sentence, tf_idf_score, lower_threshold, upper_threshold))
    df = pd.DataFrame()
    df["sentence"] = list_of_sentences
    df["keywords"] = key_list
    df.to_csv("keywords_ds.csv")
    df.to_pickle("keywords_ds.pkl")
