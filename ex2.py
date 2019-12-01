import nltk
import pandas as pd
# nltk.download('brown')
from nltk.corpus import brown
from collections import defaultdict, Counter
import operator
import probabilities

START, STOP = "START", "STOP"
DYNAMIC_STOP = "*"

def generate_dictionary(sentences):
    """Generate dictionary: {word_type: {POS:counter}}
    sentences is a list of sentences, each sentence is list of (word, POS)
    tuples."""
    word_tags = defaultdict(Counter)
    for sentence in sentences:
        for word, pos in sentence:
            word_tags[word][pos] +=1
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

def get_accuracy(test_set,max_pos_train):
    true_positive = 0
    total_num_of_words = 0
    for sentence in test_set:
        for word, pos in sentence:
            if pos == max_pos_train[word]:
                true_positive += 1
            total_num_of_words += 1
    return (true_positive/total_num_of_words)


def calc_test_error(test_set, max_pos_train):
    """Clculate 1-accuracy"""
    max_pos_train = add_unknowns(test_set,max_pos_train)
    err_rate = 1 - get_accuracy(test_set,max_pos_train)
    return err_rate

def Qb(train_news, test_news):
    train_dict = generate_dictionary(train_news)
    max_pos_train = get_mle_train(train_dict)
    error = calc_test_error(test_news, max_pos_train)
    print(error)





def pad_training_set(train_set):
    """pad training set with START values"""
    for sentence in train_set:
        sentence.insert(0, (START, START))


#
# def arg_max(k, sentence,dynamic_table):
#     pi(k,sentence[k-1], dynamic_table)
#
# def vitarti(sentence):
#     """
#     :param sentence: THE INPUT SEMTENCE HAS START AND STOP
#     :return:
#     """
#     n = len(sentence)
#     dynamic_table = [None]* n
#     dynamic_table[0] = 1
#     for k in range(1,n):
#         max_estimate = 0
#         for t in avaliable_tags_for_v(v):
#             cur_pi = pi(k,t)
#
#             if cur_pi > max_estimate:
#                 max_pi = cur_pi
#
#         dynamic_table[k] = arg_max(k, sentence[k-1],dynamic_table)

def init_word_set(S):
    dict = {}
    for tag in S:
        dict[tag] = 0
    return dict


# def pi(k, v, S, x, train_set):
#     if k == 1: return 1
#     word_set = init_word_set(S)
#     for w in S:
#         word_set[w] = pi(k-1, w, S, x, train_set) * trans_prob(v, w, S) * emission_prob(x[k], v, train_set)
#
#     return max(word_set.items(), key=operator.itemgetter(1))[0]     # return max key by value in word_set

# def viterbi(x, S, train_set):
#     pi_table = [[]]
#     n = len(x)
#     for k in range(n):
#         for v in S:
#             pi_table[k][v] = pi(k,v,S,x, train_set=train_set)
#     possible_tags = init_word_set(S)
#     for v in S:
#         possible_tags[v] = pi(n, v, S, x, train_set) * trans_prob(STOP, v, S)
#
#     return max(possible_tags.items(), key=operator.itemgetter(1))[0]    # return max key by value in word_set


def create_viterbi_table(x, probs):
    """

    :param x: Sentence x1-xn
    :param S: All possible tags
    :param train_set: list of sentences
    :return: Best next pos
    """
    pi = []

    print('length of S is ')
    print(probs.S_len)
    pi.append([1]*probs.S_len)
    for k in range(1 ,len(x)):
        pi.append([(0,0)]*probs.S_len)
        print('k=' + str(k))
        for j in range(probs.S_len):
            print("J = "+ str(j))
            max_index = None
            max_value = 0
            e = probs.emission_table[k][j]
            for i in range(probs.S_len):
                cur = pi[k-1][i] * probs.emission_table[j][i] * e
                if cur > max_value:
                    max_value = cur
                    max_index = i
            pi[k][j] = (max_value, max_index)
    return pi

def viterbi(x,S,train_set):
    probs = probabilities.Probabilities(S)
    probs.generate_emission_table(x, train_set)
    probs.generate_transition_table(train_set)
    print("viterbi")
    pi = create_viterbi_table(x, probs)
    print(pi)
    tag_vec = []
    max_prob = 0
    best_index = 0
    k = -1
    for i in pi[k]:
        prob, previous_ind = pi[k][i]
        if prob > max_prob:
            best_index = i
            max_prob = prob

    tag_vec.append(S[best_index])
    previous_ind = pi[k][best_index][1]
    k -= 1
    while -k < len(x):
        print("while")
        tag_vec.append(S[previous_ind])
        previous_ind = pi[k][previous_ind][1]
        k -= 1

    tag_vec = tag_vec.reverse()
    return tag_vec



def get_all_tags(sentences):
    """returns a dictionary of all tags per a certain word"""
    word_tags = defaultdict(set)
    for sentence in sentences:
        for word, pos in sentence:
            word_tags[word].add(pos)
    return word_tags

def initialize_S(train_set):
#     add from test set with unknown tag as well??
    S = set()
    for sentence in train_set:
        for word, pos in sentence:
            S.add(pos)
    return S


def Qc(train_set, test_set):
    S = initialize_S(train_set)     #Rois version
    # S = get_all_tags(train_set)   #Tomers version
    viterbi_res= []
    for sentence in test_set:
        x = [t[0] for t in sentence]
        viterbi_tags = viterbi(x, S, train_set)
        print(viterbi_tags)
        viterbi_res.append(viterbi_tags)



def main():
    tagged_news = (brown.tagged_sents(categories='news'))
    threshold = int(len(tagged_news) * 0.1)
    train_news = tagged_news[:-threshold]
    test_news = tagged_news[-threshold:]

    # a = get_all_tags(train_news)
    # Qb(train_news, test_news)
    Qc(train_set=train_news, test_set=test_news)


if __name__ == '__main__':
    main()

# df = pd.DataFrame(news,)


