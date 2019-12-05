from collections import defaultdict, Counter

START = "START"
UNKNOWN = 'NN'


class Probabilities():
    """ A class that contains and manages emission and transition 
    probability dictionaries.
    """

    def __init__(self, S, train_set, test_set):
        S.add(UNKNOWN)
        self.S_list = list(S)
        self.S_len = len(self.S_list)

        # {word: True/False}
        self.existing_words = {}

        ### Emission dictionaries###

        # {(word,pos): count}
        self.d1 = defaultdict(int)

        # {pos : count}
        self.d2 = defaultdict(int)

        ### Transition dictionaries###

        # {(pos,prev_pos): count}
        self.d3 = defaultdict(int)

        # {pos: count}
        self.d4 = defaultdict(int)

        self.generate_existing_words(train_set, test_set)
        self.generate_pos_dic(train_set)
        self.add_unknown_words(test_set)

    def generate_pos_dic(self, train_set):
        """Fills in dictionaries 1-4 (i.e., the emission and tranition dictionaries)"""
        for xy_tups in train_set:
            for i in range(len(xy_tups)):
                word = xy_tups[i][0]
                pos = xy_tups[i][1]
                self.d1[(word, pos)] += 1
                self.d2[pos] += 1
                if i != 1:
                    prev_pos = xy_tups[i-1][1]
                    self.d3[(pos, prev_pos)] += 1
                    self.d4[pos] += 1

    def generate_existing_words(self, train_set, test_set):
        """Generate the dictionary of all words in the train and test sets.
        At first add false values to all words in test set.
        Then, add True values to all words in train set.
        NOTE if word is in test and train set both then it gets the value True.
        Arguments:
            train_set 
            test_set 
        """
        for xy_tups in test_set:
            for tup in xy_tups:
                self.existing_words[tup[0]] = False

        for xy_tups in train_set:
            for tup in xy_tups:
                self.existing_words[tup[0]] = True

    def add_unknown_words(self, test_set):
        """ Adds words exclusively in the test set to the d1 dictionary

        Arguments:
            test_set
        """
        for xy_tups in test_set:
            for tup in xy_tups:
                if self.existing_words[tup[0]] == False:
                    self.d1[(tup[0], UNKNOWN)] += 1

    def e(self, x, y):
        """
        :param x: word
        :param y: POS
        :return:
        """
        sum_xy = self.d1[(x, y)]
        sum_y = self.d2[y]
        return float(sum_xy/sum_y)

    def q(self, y1, y2):
        """
        :param y1: The POS event
        :param y2: The POS condition
        :return:
        """
        if self.d3[(y1, y2)] == 0:
            return 0
        return float(self.d3[(y1, y2)]/self.d4[y1])
