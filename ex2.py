import re
import nltk
# nltk.download('brown')
import statistics
from nltk.corpus import brown
from collections import defaultdict, Counter
from pandas import *
from probabilities import Probabilities
from matplotlib import pyplot as plt
import numpy as np


START = "START"

################# QUESTION b##################################


def generate_dictionary(sentences):
    """Generate dictionary: {word_type: {POS:counter}}
    sentences is a list of sentences, each sentence is list of (word, POS)
    tuples."""
    word_tags = defaultdict(Counter)
    for sentence in sentences:
        for word, pos in sentence:
            word_tags[word][pos] += 1
    return word_tags


def get_mle_train(words_dict):
    max_pos_train = {}
    for word in words_dict:
        max_pos_train[word] = words_dict[word].most_common(1)[0][0]
    return max_pos_train


def add_unknowns(test_set, max_pos_train):
    unknown_set = set()
    for sentence in test_set:
        for word, pos in sentence:
            if word in max_pos_train.keys():
                continue
            unknown_set.add(word)
            max_pos_train[word] = 'NN'
    return max_pos_train, unknown_set


def get_accuracy_B(test_set, max_pos_train, unknown_set):
    unknown_0 = False
    known_0 = False
    known_positive = 0
    total_known = 0
    unknown_positive = 0
    total_unknown = 0
    true_positive = 0
    total_num_of_words = 0
    for sentence in test_set:
        for word, pos in sentence:
            good_tag = 0
            if pos == max_pos_train[word]:
                true_positive += 1
                good_tag = 1
            if word not in unknown_set: #if known
                total_known += 1
                known_positive += good_tag
            else:
                total_unknown += 1
                unknown_positive += good_tag
            total_num_of_words += 1

    # if total_unknown == 0:
    #     total_unknown = 1
    #     unknown_0 = True
    # if total_known == 0:
    #     total_known = 1
    #     known_0 = True

    gen_acc = true_positive/total_num_of_words
    known_acc = known_positive/total_known
    unknown_acc = unknown_positive/total_unknown
    return [gen_acc, known_acc, unknown_acc]


def calc_test_error(test_set, max_pos_train):
    """Calculate 1-accuracy"""
    max_pos_train , unknowns_set = add_unknowns(test_set, max_pos_train)
    all_acc = get_accuracy_B(test_set, max_pos_train,unknowns_set)
    gen_error = 1 - all_acc[0]
    known_error = 1- all_acc[1]
    unknown_error = 1 - all_acc[2]
    return [gen_error, known_error, unknown_error]


def Qb(train_news, test_news):
    train_dict = generate_dictionary(train_news)
    max_pos_train = get_mle_train(train_dict)
    err_vector = calc_test_error(test_news, max_pos_train)
    return err_vector


################# QUESTION C ##################################


def pad_training_set(train_set):
    """pad training set with START values"""
    for sentence in train_set:
        sentence.insert(0, (START, START))


def init_word_set(S):
    dict = {}
    for tag in S:
        dict[tag] = 0
    return dict


def create_viterbi_table(x, probs, laplace=False):
    """
    :param x: Sentence x1-xn
    :param probs: probability object
    :return: Best next pos
    pi tuples: (value, index)
    """
    pi = []
    pi.append([(1, 0)]*probs.S_len)
    for k in range(1, len(x)):
        pi.append([(0, 0)]*probs.S_len)
        for v_index in range(probs.S_len):
            max_index = 0
            max_value = 0
            e = probs.e(x[k], probs.S_list[v_index], laplace)
            for w_index in range(probs.S_len):
                q = probs.q(probs.S_list[v_index], probs.S_list[w_index])
                cur = pi[k-1][w_index][0] * q * e
                if cur > max_value:
                    max_value = cur
                    max_index = w_index  # means i is the best tag!
            pi[k][v_index] = (max_value, max_index)
    return pi


def viterbi(x, probs, laplace=False):
    """ A function that runs the Viterbi algorithm.

    Arguments:
        x {[List]} -- a list containing all the sentences in the test set.

    Returns:
        [List] -- a vector (python list) of the POS tags that are the prediction of 
        the viterbi algorithm for the sentence x.
    """
    pi = create_viterbi_table(x, probs, laplace)
    tag_vec = []
    max_prob = 0
    best_index = 0
    k = -1
    for i in range(len(pi[k])):  # starts at the -1 row!
        prob, previous_ind = pi[k][i]  # finds best probabillity there
        if prob > max_prob:
            best_index = i
            max_prob = prob

    # traverse the pi table
    tag_vec.append(probs.S_list[best_index])
    previous_ind = pi[k][best_index][1]
    k -= 1
    while -k < len(x):
        tag_vec.append(probs.S_list[previous_ind])
        previous_ind = pi[k][previous_ind][1]
        k -= 1

    tag_vec = tag_vec[::-1]
    return tag_vec


def get_all_tags(sentences):
    """returns a dictionary of all tags per a certain word"""
    word_tags = defaultdict(set)
    for sentence in sentences:
        for word, pos in sentence:
            word_tags[word].add(pos)
    return word_tags


def initialize_S(train_set):
    """ Initialize the set of all tags in the training set.
    Arguments:
        train_set {[type]} -- the training set.
    """
    S = set()
    for sentence in train_set:
        for tup in sentence:
            S.add(tup[1])
    return S


def calculate_error(results, y, sent, probs, qe = False):
    """ Calculate the error between the results and the actual tags

    Arguments:
        results {[type]} -- [description]
        y {[type]} -- [description]
    """
    known_0 = False
    unknown_0 = False
    total_correct = 0
    correct_answers = 0
    known_correct = 0
    unknowns_correct = 0
    total_words = 0
    total_known = 0
    total_unknown = 0

    if (len(y) > len(results)):
        results.insert(0, "AP")
    if (len(y) > len(results)):
        # TODO add most common tag to the beggining instead of AP
        results.insert(0, "AP")
    for i in range(len(results)):
        good_tag = 0
        if results[i] == y[i]:
            correct_answers += 1
            good_tag = 1
        if not qe:
            if probs.existing_words[sent[i]]: #if known
                total_known += 1
                known_correct += good_tag
            else:
                total_unknown += 1
                unknowns_correct += good_tag

    if total_known == 0:
        total_known = 1
        known_0 = True
    if total_unknown ==0:
        total_unknown = 1
        unknown_0 = True

    gen_error = 1 - float(correct_answers/len(results))
    known_error = 1 - float(known_correct/total_known)
    unknown_err = 1 - float(unknowns_correct/total_unknown)

    return [gen_error, known_error, unknown_err], known_0, unknown_0


def clean_POS(sentence_set):
    """updates complex POS tags to be the prefix of the original.
    e.g. PPS+BEZ -> PPS

    Arguments:
        sentence_set {[list]} -- a set of tagged sentences (usually 
        either the training or test set)

    Returns:
        [list] -- a "cleaned" set of sentences
    """
    clean_set = []
    for xy_tups in sentence_set:
        clean_xy_tups = []
        for word, pos in xy_tups:
            pos = re.split('\+|\-', pos)[0]
            clean_xy_tups.append((word, pos))
        clean_set.append(clean_xy_tups)
    return clean_set


def Qc(train_set, test_set, laplace=False):
    """Handles the tasks of question c

    Arguments:
        train_set
        test_set 

    Keyword Arguments:
        laplace {bool} -- are we to use Laplace smoothing or not (default: {False})
    """
    gen_error_vec = []
    known_error_vec = []
    unknown_error_vec = []

    viterbi_results = []
    train_set = clean_POS(train_set)
    test_set = clean_POS(test_set)
    S = initialize_S(train_set)
    probs = Probabilities(S, train_set=train_set, test_set=test_set)
    for xy_tup in test_set:
        x = [t[0] for t in xy_tup]
        y = [t[1] for t in xy_tup]
        viterbi_tags = viterbi(x, probs, laplace)
        viterbi_results.append(viterbi_tags)
        err_vec, known_0, unknonwn_0 = (calculate_error(viterbi_tags, y, x, probs))
        gen_error_vec.append(err_vec[0])
        if not known_0: known_error_vec.append(err_vec[1])
        if not unknonwn_0: unknown_error_vec.append(err_vec[2])
    gen_error = statistics.mean(gen_error_vec)
    known_error = statistics.mean(known_error_vec)
    unknown_error = statistics.mean(unknown_error_vec)
    return [gen_error, known_error, unknown_error]

##################### QUESTION D ##################################


def Qd(train_set, test_set):
    """Simply run Qc (viterbi) with laplace smoothing
    """
    return Qc(train_set, test_set, True)

###################################################################


def confusion_matrix(S, pseudo_probs):
    """create the confusion matrix 

    Arguments:
        S {[set]} -- a set of tags including the pseudo tags
        pseudo_probs {[type]} -- [description]

    Returns:
        [2D array] -- the confusion matrix
    """
    S = list(S)
    confusion = []
    for i in range(len(S)):
        row = []
        for j in range(len(S)):
            row.append(pseudo_probs.get_confusion_value(
                S[i], S[j]))
        confusion.append(row)
    return confusion


def Qe(train_set, test_set, laplace=False):
    """Handles tasks of question e

    Arguments:
        train_set 
        test_set 

    Keyword Arguments:
        laplace {bool} -- (default: {False})
    """
    # initializations
    viterbi_results = []
    gen_error_vec = []
    known_error_vec = []
    unknown_error_vec = []


# "clean" the train and test sets from complex tags
    train_set = clean_POS(train_set)
    test_set = clean_POS(test_set)

    S = initialize_S(train_set)
    probs = Probabilities(S, train_set, test_set)
    # Generate pseudo train and test sets and probability object
    pseudo_train = probs.generate_pseudo_set(train_set)
    pseudo_test = probs.generate_pseudo_set(test_set)
    pseudo_probs = Probabilities(S, pseudo_train, pseudo_test)
    for xy_tup in pseudo_test:
        x = [t[0] for t in xy_tup]
        y = [t[1] for t in xy_tup]
        viterbi_tags = viterbi(x, pseudo_probs, laplace)
        viterbi_results.append(viterbi_tags)
        err_vec, _, _ = (calculate_error(viterbi_tags, y, x, probs, True))
        gen_error_vec.append(err_vec[0])
        # update confusion values
        pseudo_probs.update_confusion_matrix(y, viterbi_tags)
    gen_error = statistics.mean(gen_error_vec)
    print(gen_error)
    # print results and statistics
    if laplace:
        print(DataFrame(confusion_matrix(S, pseudo_probs)))
    return gen_error


def Qe_Laplace(train_set, test_set):
    return Qe(train_set, test_set, True)

####################################################################

def plot_graphs(train_set,test_set):
    errs_for_first_3 = ('General Error','Known Words error','Unknown words error')
    plot_errors(train_set, test_set, "MLE estimation", Qb, errs_for_first_3)
    plot_errors(train_set, test_set, "Viterbi+ HMM", Qc, errs_for_first_3)
    plot_errors(train_set, test_set, "Viterbi+ Laplace", Qd, errs_for_first_3)
    plot_errors(train_set, test_set, "Pseudo-words tagging", Qe, ['General Error'])
    plot_errors(train_set, test_set, "Pseudo-words + Laplace", Qe_Laplace, ['General error'])


def plot_errors(train_set, test_set, name, func, errs):
    y_pos = np.arange(len(errs))
    performance = func(train_set,test_set)
    plt.bar(y_pos,performance, align='center', alpha = 0.5)
    plt.xticks(y_pos, errs)
    plt.ylabel("Error rate")
    plt.title(name)
    plt.show()


def main():
    tagged_news = (brown.tagged_sents(categories='news'))
    threshold = int(len(tagged_news) * 0.1)
    train_news = tagged_news[:-threshold]
    test_news = tagged_news[-threshold:]
    plot_graphs(train_news, test_news)


if __name__ == '__main__':
    main()
