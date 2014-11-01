# Cooccurrence matrix construction tools
# for fitting the GloVe model.

try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

from .corpus_cython import construct_cooccurrence_matrix


class Corpus(object):
    """
    Class for constructing a cooccurrence matrix
    from a corpus.
    """
    
    def __init__(self):

        self.dictionary = None
        self.matrix = None

    def fit(self, corpus, window=10):
        """
        Perform a pass through the corpus to construct
        the cooccurrence matrix. 

        You must call fit_dictionary first.

        Parameters:
        - iterable of lists of strings corpus
        - int window: the length of the (symmetric)
          context window used for cooccurrence.
        """

        self.dictionary, self.matrix = construct_cooccurrence_matrix(corpus, 
                                                                     int(window))

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
