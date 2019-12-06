import re
import nltk
# nltk.download('brown')
import statistics
from nltk.corpus import brown
from collections import defaultdict, Counter
from pandas import *
from probabilities import Probabilities

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
    for sentence in test_set:
        for word, pos in sentence:
            if word in max_pos_train.keys():
                continue
            max_pos_train[word] = 'NN'
    return max_pos_train


def get_accuracy(test_set, max_pos_train):
    true_positive = 0
    total_num_of_words = 0
    for sentence in test_set:
        for word, pos in sentence:
            if pos == max_pos_train[word]:
                true_positive += 1
            total_num_of_words += 1
    return (true_positive/total_num_of_words)


def calc_test_error(test_set, max_pos_train):
    """Calculate 1-accuracy"""
    max_pos_train = add_unknowns(test_set, max_pos_train)
    err_rate = 1 - get_accuracy(test_set, max_pos_train)
    return err_rate


def Qb(train_news, test_news):
    train_dict = generate_dictionary(train_news)
    max_pos_train = get_mle_train(train_dict)
    error = calc_test_error(test_news, max_pos_train)
    print(error)


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


def create_viterbi_table(x, probs, laplace = False):
    """
    :param x: Sentence x1-xn
    :param probs: probability object
    :return: Best next pos
    pi tuples: (value, index)
    """
    pi = []
    # add a row of values 1 as the first row
    pi.append([(1, 0)]*probs.S_len)
    for k in range(1, len(x)): #0 to 1
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


def viterbi(x, probs, laplace = False):
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
#   TODO  add from test set with unknown tag as well??
    S = set()
    for sentence in train_set:
        for tup in sentence:
            S.add(tup[1])
    return S

def calculate_error(results, y):
    """ Calculate the error between the results and the actual tags
    
    Arguments:
        results {[type]} -- [description]
        y {[type]} -- [description]
    """
    correct_answers = 0
    # print("y is bigger in: " + str(len(y)- len(results)))
    if (len(y)> len(results)):
        results.insert(0,"AP")
    if (len(y) > len(results)):
        results.insert(0, "AP") #  TODO add most common tag to the beggining instead of AP
    for i in range(len(results)):
        if results[i] == y[i]:
            correct_answers += 1
    return 1 - float(correct_answers/len(results))


def clean_POS(sentence_set):
    """updates complex POS tags to be the prefix of the original.
    e.g. PPS+BEZ -> PPS

    Arguments:
        xy_tups {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    clean_set = []
    for xy_tups in sentence_set:
        clean_xy_tups = []
        for word, pos in xy_tups:
            pos = re.split('\+|\-', pos)[0]
            clean_xy_tups.append((word, pos))
        clean_set.append(clean_xy_tups)
    return clean_set

def Qc(train_set, test_set, laplace = False):
    viterbi_results = []
    errors = []
    train_set = clean_POS(train_set)
    test_set = clean_POS(test_set)
    S = initialize_S(train_set)  # Rois version
    probs = Probabilities(S, train_set=train_set, test_set=test_set)
    for xy_tup in test_set:
        x = [t[0] for t in xy_tup]
        y = [t[1] for t in xy_tup]
        viterbi_tags = viterbi(x, probs, laplace)
        viterbi_results.append(viterbi_tags)
        errors.append(calculate_error(viterbi_tags, y))
    print(errors)
    print(statistics.mean(errors))

###################################################################
def Qd(train_set, test_set):
    Qc(train_set, test_set, True)

###################################################################

# def confusion_matrix(d_true_tags, d_predicted_tags, probs):
#     confusion = []
#     for i in range(len(d_true_tags)):
#         row = []
#         for j in range(len(d_predicted_tags)):
#             row.append(probs.get_confusion_value(d_true_tags[i], d_predicted_tags[j]))
#         confusion.append(row)
#     return confusion

def confusion_matrix(pseudo_S, pseudo_probs):
    pseudo_S = list(pseudo_S)
    confusion = []
    for i in range(len(pseudo_S)):
        row = []
        for j in range(len(pseudo_S)):
            row.append(pseudo_probs.get_confusion_value(pseudo_S[i], pseudo_S[j]))
        confusion.append(row)
    return confusion


def Qe(train_set, test_set, laplace = False):
    viterbi_results = []
    errors = []
    S = initialize_S(train_set)
    probs = Probabilities(S, train_set,test_set)
    train_set = clean_POS(train_set)
    test_set = clean_POS(test_set)
    pseudo_train = probs.generate_pseudo_set(train_set)
    pseudo_test = probs.generate_pseudo_set(test_set)

    pseudo_S = initialize_S(pseudo_train)
    pseudo_probs = Probabilities(pseudo_S, pseudo_train, pseudo_test)
    for xy_tup in pseudo_test:
        x = [t[0] for t in xy_tup]
        y = [t[1] for t in xy_tup]
        viterbi_tags = viterbi(x, pseudo_probs, laplace)
        viterbi_results.append(viterbi_tags)
        errors.append(calculate_error(viterbi_tags, y))
        # update confusion values
        pseudo_probs.update_confusion_matrix(y, viterbi_tags)
    print(errors)
    print(statistics.mean(errors))
    print(DataFrame(confusion_matrix(pseudo_S, pseudo_probs)))


####################################################################
def Qe_Laplace(train_set,test_set):
    Qe(train_set,test_set, True)

def main():
    tagged_news = (brown.tagged_sents(categories='news'))
    threshold = int(len(tagged_news) * 0.1)
    train_news = tagged_news[:-threshold]
    test_news = tagged_news[-threshold:]
    # print(train_news)
    # a = get_all_tags(train_news)
    # Qb(train_news, test_news)
    # Qd(train_set=train_news, test_set=test_news)
    # Qd(train_news, test_news)
    # Qe(train_news,test_news)
    Qe_Laplace(train_news, test_news)


if __name__ == '__main__':
    main()
