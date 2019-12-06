from collections import defaultdict

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

        # {pos : count} takes into account the first word of a sentence
        self.d2 = defaultdict(int)

        ### Transition dictionaries###

        # {(pos,prev_pos): count}
        self.d3 = defaultdict(int)

        # {pos: count} missing first words of each sentence
        self.d4 = defaultdict(int)

        self.distinct_words = set()

        # non exisiting words or words that apeared less than threshold
        self.low_frequency_words = set()

        # count of each word
        self.word_count = defaultdict(int)
        self.generate_existing_words(train_set, test_set)
        self.generate_dictionaries(train_set)
        self.add_unknown_words(test_set)
        self.generate_low_freq_dictionary()
        self.delta = 1
        self.confusion_dict = defaultdict(int)

    def generate_dictionaries(self, train_set):
        """Fills in dictionaries 1-4 (i.e., the emission and tranition dictionaries)"""
        for xy_tups in train_set:
            for i in range(len(xy_tups)):
                word = xy_tups[i][0]
                pos = xy_tups[i][1]
                self.word_count[word] += 1
                self.distinct_words.add(word)
                self.d1[(word, pos)] += 1
                self.d2[pos] += 1
                if i != 1:
                    prev_pos = xy_tups[i - 1][1]
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

    def generate_low_freq_dictionary(self, thres=3):
        for word in self.existing_words:
            if self.existing_words[word] == False:
                self.low_frequency_words.add(word)
            elif self.word_count[word] <= thres:
                self.low_frequency_words.add(word)

    def add_unknown_words(self, test_set):
        """ Adds words exclusively in the test set to the d1 dictionary

        Arguments:
            test_set
        """
        for xy_tups in test_set:
            for tup in xy_tups:
                if self.existing_words[tup[0]] == False:
                    # print(self.existing_words)
                    self.d1[(tup[0], UNKNOWN)] += 1

    def e(self, x, y, laplace=False):
        """
        :param x: word
        :param y: POS
        :return:
        """
        sum_xy = self.d1[(x, y)]
        sum_y = self.d2[y]
        if sum_xy == 0 and not laplace:
            return 0
        if laplace:
            sum_xy += self.delta
            sum_y += self.delta * len(self.distinct_words)
        return float(sum_xy / sum_y)

    def q(self, y1, y2):
        """
        :param y1: The POS event
        :param y2: The POS condition
        :return:
        """
        if self.d3[(y1, y2)] == 0:
            return 0
        return float(self.d3[(y1, y2)] / self.d4[y1])

    def tag_word(self, word, index_in_sent):
        """
        Generate pseudo word dictionary for every low_frequency word"""
        if (word[0].isupper()):
            return "InitCap"
        if word[0].isdigit():
            return "DigitAndLetters"
        if index_in_sent == 0:
            return "FirstWord"
        return "Other"

    def generate_pseudo_set(self, t_set):
        """
        Assumptions: t_set is of the form:
        list of lists of tuples (word,tag)
        :param t_set:
        :return: new set where each frequent word is replaced by its pseudo word.
        """
        new_set = []
        for xy_tups in t_set:
            new_sent = []
            for i, tup in enumerate(xy_tups):
                word = xy_tups[i][0]
                pos = xy_tups[i][1]
                if (word in self.low_frequency_words) or self.existing_words[word] is False:
                    word = self.tag_word(word, i)
                new_sent.append((word, pos))
            new_set.append(new_sent)

        return new_set

    def update_confusion_matrix(self, true_tags, predicted_tags):
        assert len(true_tags) == len(predicted_tags)

        for i in range(len(true_tags)):
            self.confusion_dict[(true_tags[i], predicted_tags[i])] += 1

    def get_confusion_value(self, tag_i, tag_j):
        return self.confusion_dict[(tag_i, tag_j)]
