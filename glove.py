# GloVe model from the NLP lab at Stanford:
# http://nlp.stanford.edu/projects/glove/.

import cPickle as pickle
import numpy as np
import scipy.sparse as sp

from glove_cython import fit_vectors


class Glove(object):
    """
    Class for estimating GloVe word embeddings using the
    corpus coocurrence matrix.
    """

    def __init__(self, no_components=30, learning_rate=0.05,
                 alpha=0.75, max_count=50):
        """
        Parameters:
        - int no_components: number of latent dimensions
        - float learning_rate: learning rate for SGD estimation.
        - float alpha, float max_count: parameters for the 
          weighting function (see the paper).
        """
        
        self.no_components = no_components
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.max_count = float(max_count)

        self.word_vectors = None
        self.word_biases = None

        self.dictionary = None
        self.inverse_dictionary = None

    def fit(self, matrix, epochs=5, no_threads=2, verbose=False):
        """
        Estimate the word embeddings.

        Parameters:
        - scipy.sparse.coo_matrix matrix: coocurrence matrix
        - int epochs: number of training epochs
        - int no_threads: number of training threads
        - bool verbose: print progress messages if True
        """

        shape = matrix.shape

        if (len(shape) != 2 or
            shape[0] != shape[1]):
            raise Exception('Coocurrence matrix must be square')

        if not sp.isspmatrix_coo(matrix):
            raise Exception('Coocurrence matrix must be in the COO format')

        self.word_vectors = np.random.rand(shape[0],
                                           self.no_components)
        self.word_biases = np.zeros(shape[0], 
                                    dtype=np.float64)
        shuffle_indices = np.arange(matrix.nnz, dtype=np.int32)

        if verbose:
            print ('Performing %s training epochs '
                   'with %s threads') % (epochs, no_threads)

        for epoch in xrange(epochs):

            if verbose:
                print 'Epoch %s' % epoch

            # Shuffle the coocurrence matrix
            np.random.shuffle(shuffle_indices)

            fit_vectors(self.word_vectors,
                        self.word_biases,
                        matrix.row,
                        matrix.col,
                        matrix.data,
                        shuffle_indices,
                        self.learning_rate,
                        self.max_count,
                        self.alpha,
                        int(no_threads))

    def add_dictionary(self, dictionary):
        """
        Supply a word-id dictionary to allow similarity queries.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if len(dictionary) > self.word_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller '
                            'or equal to the number of word vectors')

        self.dictionary = dictionary
        self.inverse_dictionary = {v: k for k, v in self.dictionary.iteritems()}

    def save(self, filename):
        """
        Serialize model to filename.
        """

        with open(filename, 'wb') as savefile:
            pickle.dump(self.__dict__,
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """
        Load model from filename.
        """
        
        instance = Glove()

        with open(filename, 'rb') as savefile:
            instance.__dict__ = pickle.load(savefile)

        return instance

    def most_similar(self, word, number=5):
        """
        Run a similarity query, retrieving number
        most similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')
        
        try:
            word_idx = self.dictionary[word]
        except KeyError:
            raise Exception('Word not in dictionary')

        word_vec = self.word_vectors[word_idx]
        dst = (np.dot(self.word_vectors, word_vec)
               / np.linalg.norm(self.word_vectors, axis=1))
        word_ids = np.argsort(-dst)

        return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[1:number]
                if x in self.inverse_dictionary]
