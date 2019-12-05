from collections import defaultdict, Counter

START = "START"


class Probabilities():
    """ A class that contains and manages emission and transition 
    probability dictionaries.
    """

    def __init__(self, S, train_set):
        self.trans_table = []
        self.emission_table = []
        self.S_list = list(S)
        self.S_len = len(self.S_list)

        ### Emission dictionaries###

        # d1 = {(word,pos): count}
        self.d1 = defaultdict(int)

        # d2 = {pos : count}
        self.d2 = defaultdict(int)

        ### Transition dictionaries###

        # d3 = {(pos,prev_pos): count}
        self.d3 = defaultdict(int)

        # d4 = {pos: count}
        self.d4 = defaultdict(int)

        self.generate_pos_dic(train_set)

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

    def e(self, x, y):
        """
        :param x: word
        :param y: POS
        :return:
        """
        return self.d1[(x, y)]/self.d2[y]

    def q(self, y1, y2):
        """
        :param y1: The POS event
        :param y2: The POS condition
        :return:
        """
        if self.d3[(y1, y2)] == 0:
            return 0
        return self.d3[(y1, y2)]/self.d4[y1]