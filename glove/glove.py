# GloVe model from the NLP lab at Stanford:
# http://nlp.stanford.edu/projects/glove/.

import collections
try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import scipy.sparse as sp

from .glove_cython import fit_vectors, transform_paragraph


class Glove(object):
    """
    Class for estimating GloVe word embeddings using the
    corpus coocurrence matrix.
    """

    def __init__(self, no_components=30, learning_rate=0.05,
                 alpha=0.75, max_count=100):
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

        self.vectors_sum_gradients = None
        self.biases_sum_gradients = None

        self.dictionary = None
        self.inverse_dictionary = None
        
        self.global_loss = None

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

        self.word_vectors = ((np.random.rand(shape[0],
                                             self.no_components) - 0.5)
                                             / self.no_components)
        self.word_biases = np.zeros(shape[0], 
                                    dtype=np.float64)

        self.vectors_sum_gradients = np.ones_like(self.word_vectors)
        self.biases_sum_gradients = np.ones_like(self.word_biases)

        shuffle_indices = np.arange(matrix.nnz, dtype=np.int32)

        if verbose:
            print('Performing %s training epochs '
                  'with %s threads' % (epochs, no_threads))

        for epoch in range(epochs):

            if verbose:
                print('Epoch %s' % epoch)

            # Shuffle the coocurrence matrix
            np.random.shuffle(shuffle_indices)
            
            self.global_loss = np.zeros(1,  dtype=np.float64)

            fit_vectors(self.word_vectors,
                        self.vectors_sum_gradients,
                        self.word_biases,
                        self.biases_sum_gradients,
                        matrix.row,
                        matrix.col,
                        matrix.data,
                        shuffle_indices,
                        self.learning_rate,
                        self.max_count,
                        self.alpha,
                        self.global_loss,
                        int(no_threads))
                        
            print('Global loss: %d' % self.global_loss)

            if not np.isfinite(self.word_vectors).all():
                raise Exception('Non-finite values in word vectors. '
                                'Try reducing the learning rate.')

    def transform_paragraph(self, paragraph, epochs=50, ignore_missing=False):
        """
        Transform an iterable of tokens into its vector representation
        (a paragraph vector).

        Experimental. This will return something close to a tf-idf
        weighted average of constituent token vectors by fitting 
        rare words (with low word bias values) more closely.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit to transform paragraphs')

        if self.dictionary is None:
            raise Exception('Dictionary must be provided to '
                            'transform paragraphs')

        cooccurrence = collections.defaultdict(lambda: 0.0)
            
        for token in paragraph:
            try:
                cooccurrence[self.dictionary[token]] += self.max_count / 10.0
            except KeyError:
                if not ignore_missing:
                    raise

        word_ids = np.array(cooccurrence.keys(), dtype=np.int32)
        values = np.array(cooccurrence.values(), dtype=np.float64)
        shuffle_indices = np.arange(len(word_ids), dtype=np.int32)

        # Initialize the vector to mean of constituent word vectors
        paragraph_vector = np.mean(self.word_vectors[word_ids], axis=0)
        sum_gradients = np.ones_like(paragraph_vector)

        # Shuffle the coocurrence matrix
        np.random.shuffle(shuffle_indices)
        transform_paragraph(self.word_vectors,
                            self.word_biases,
                            paragraph_vector,
                            sum_gradients,
                            word_ids,
                            values,
                            shuffle_indices,
                            self.learning_rate,
                            self.max_count,
                            self.alpha,
                            epochs)

        return paragraph_vector

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
        if hasattr(self.dictionary, 'iteritems'):
            # Python 2 compat
            items_iterator = self.dictionary.iteritems()
        else:
            items_iterator = self.dictionary.items()

        self.inverse_dictionary = {v: k for k, v in items_iterator}

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

    def _similarity_query(self, word_vec, number):

        dst = (np.dot(self.word_vectors, word_vec)
               / np.linalg.norm(self.word_vectors, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)

        return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
                if x in self.inverse_dictionary]

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

        return self._similarity_query(self.word_vectors[word_idx], number)[1:]

    def most_similar_paragraph(self, paragraph, number=5, **kwargs):
        """
        Return words most similar to a given paragraph (iterable of tokens).
        """

        paragraph_vector = self.transform_paragraph(paragraph, **kwargs)

        return self._similarity_query(paragraph_vector, number)
