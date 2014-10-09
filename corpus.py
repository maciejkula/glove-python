# Cooccurrence matrix construction tools
# for fitting the GloVe model.

import cPickle as pickle

from corpus_cython import construct_cooccurrence_matrix


class Corpus(object):
    """
    Class for constructing a cooccurrence matrix
    from a corpus.
    """
    
    def __init__(self):

        self.dictionary = None
        self.matrix = None

    def fit_dictionary(self, corpus):
        """
        Perform a pass through the corpus to
        construct a word-id mapping.
        
        Parameters:
        - iterable of lists of strings corpus
        """

        dictionary = {}

        # First pass to construct the dictionary
        for context in corpus:
            for word in context:
                if word not in dictionary:
                    dictionary[word] = len(dictionary)

        self.dictionary = dictionary

    def fit_matrix(self, corpus, window=10):
        """
        Perform a pass through the corpus to construct
        the cooccurrence matrix. 

        You must call fit_dictionary first.

        Parameters:
        - iterable of lists of strings corpus
        - int window: the length of the (symmetric)
          context window used for cooccurrence.
        """
        if self.dictionary is None:
            raise Exception('You must fit the dictionary before transforming the corpus')

        self.matrix = construct_coocurrence_matrix(corpus, 
                                                   self.dictionary, int(window))

    def save(self, filename):
        
        with open(filename, 'wb') as savefile:
            pickle.dump((self.dictionary, self.matrix),
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):

        instance = cls()

        with open(filename, 'rb') as savefile:
            instance.dictionary, instance.matrix = pickle.load(savefile)

        return instance
