from collections import defaultdict, Counter

START = "START"

class Probabilities():

    def __init__(self, S, train_set):
        self.trans_table = []
        self.emission_table = []
        self.S_list = list(S)
        self.S_len = len(self.S_list)

        ### emission dictionaries###
        # d1 = {(word,pos): count}
        self.d1 = defaultdict(int)
        # d2 = {pos : count}
        self.d2 = defaultdict(int)

        ### Transition dictionaries###
        # d3 = {(pos,prev_pos): count}
        self.d3 = defaultdict(int)
        # d4 = {pos: count}
        self.d4= defaultdict(int)

        self.POS_dic = self.generate_pos_dic(train_set)

    def generate_pos_dic(self, train_set):
        """Fills in dictionaries 1-4 (i.e., the emission and tranition dictionaries)"""
        for xy_tups in train_set:
            for i in range(len(xy_tups)):
                word = xy_tups[i][0]
                pos = xy_tups[i][1]
                self.d1[(word,pos)] +=1
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


    # def emission_prob(self, x, y, set, add_one=False):
    #     """
    #     :param word:
    #     :param tag:
    #     :return: the emission probability
    #     """
    #     numerator = 0
    #     denom = 0
    #     for sentence in set:
    #         for word, tag in sentence:
    #             if tag == y:
    #                 if word == x:
    #                     numerator += 1
    #                 denom += 1
    #     # if add_one: TODO
    #     # return float((numerator + 1)/(denom + get_delta()))
    #     return float(numerator/denom)


    # def trans_prob(self, cur_tag, prev_tag, sentence_set):
    #     """
    #     run over all pairs of 2 consecutive tags in corpus
    #     :param cur_tag:
    #     :param prev_tag:
    #     :return:
    #     """
    #     numerator = 0
    #     denom = 0
    #     prev = START
    #     for sentence in sentence_set:
    #         for word, tag in sentence:
    #             if word == START:
    #                 continue
    #             if tag == cur_tag:
    #                 if prev_tag == prev:
    #                     numerator += 1
    #                 denom += 1
    #             prev_tag = tag
    #     return float(numerator/denom)

    #
    # def generate_transition_table(self, train_set):
    #     for i in range(self.S_len):
    #         self.trans_table.append([0]*self.S_len)
    #         for j in range(self.S_len):
    #             self.trans_table[i][j] = self.trans_prob(self.S_list[i], self.S_list[j], train_set)
    #     print(self.trans_table)
    #
    # def generate_emission_table(self, x, train_set):
    #     for i in range(len(x)):
    #         self.emission_table.append([0]*self.S_len)
    #         for j in range(self.S_len):
    #             self.emission_table[i][j] = self.emission_prob(x[i], self.S_list[j], train_set)
    #     print(self.emission_table)
