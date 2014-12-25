# Cooccurrence matrix construction tools
# for fitting the GloVe model.
import numpy as np
import scipy.sparse as sp

try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

from .corpus_cython import construct_cooccurrence_matrix, COOMatrix


class Corpus(object):
    """
    Class for constructing a cooccurrence matrix
    from a corpus.

    A dictionary mapping words to ids can optionally
    be supplied. If left None, it will be constructed
    from the corpus.
    """
    
    def __init__(self, dictionary=None, memmap_prefix=None):

        self.dictionary = {}
        self.dictionary_supplied = False
        self.memmapped = False
        self.matrix = None

        if dictionary is not None:
            self._check_dict(dictionary)
            self.dictionary = dictionary
            self.dictionary_supplied = True

        if memmap_prefix is not None:
            self.memmapped = True
            self.row_fname = memmap_prefix + '_rows.mmap'
            self.col_fname = memmap_prefix + '_cols.mmap'
            self.data_fname = memmap_prefix + '_data.mmap'
        else:
            self.row_fname = None
            self.col_fname = None
            self.data_fname = None

    def _check_dict(self, dictionary):

        if (np.max(list(dictionary.values())) != (len(dictionary) - 1)):
            raise Exception('The largest id in the dictionary '
                            'should be equal to its length minus one.')

        if np.min(list(dictionary.values())) != 0:
            raise Exception('Dictionary ids should start at zero')

    def fit(self, corpus, window=10, max_map_size=1000000, ignore_missing=False):
        """
        Perform a pass through the corpus to construct
        the cooccurrence matrix. 

        Parameters:
        - iterable of lists of strings corpus
        - int window: the length of the (symmetric)
          context window used for cooccurrence.
        - int max_map_size: the maximum size of map-based row storage.
                            When exceeded a row will be converted to
                            more efficient array storage. Setting this
                            to a higher value will increase speed at
                            the expense of higher memory usage.
        - bool ignore_missing: whether to ignore words missing from
                               the dictionary (if it was supplied).
                               Context window distances will be preserved
                               even if out-of-vocabulary words are
                               ignored.
                               If False, a KeyError is raised.
        """

        coo_matrix = COOMatrix(self.row_fname,
                               self.col_fname,
                               self.data_fname)

        construct_cooccurrence_matrix(corpus,
                                      self.dictionary,
                                      coo_matrix,
                                      int(self.dictionary_supplied),
                                      int(window),
                                      int(ignore_missing),
                                      max_map_size)

        self.matrix = sp.coo_matrix((coo_matrix.data_arr,
                                     (coo_matrix.rows_arr, coo_matrix.cols_arr)),
                                    shape=(len(self.dictionary),
                                           len(self.dictionary)),
                                    dtype=np.float32)

    def save(self, filename):
        
        with open(filename, 'wb') as savefile:
            if not self.memmapped:
                pickle.dump(self.__dict__,
                            savefile,
                            protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # Don't try to pickle memmapped files.
                dct = {k: v for k, v
                       in self.__dict__.items()
                       if k != 'matrix'}
                pickle.dump(dct,
                            savefile,
                            protocol=0)

    @classmethod
    def load(cls, filename):

        instance = cls()

        with open(filename, 'rb') as savefile:
            instance.__dict__ = pickle.load(savefile)

        if instance.memmapped:
            mode = 'r+'
            rows = np.memmap(instance.row_fname, dtype=np.int32,
                                  mode=mode, offset=np.int32().itemsize)
            cols = np.memmap(instance.col_fname, dtype=np.int32,
                                  mode=mode, offset=np.int32().itemsize)
            data = np.memmap(instance.data_fname, dtype=np.float32,
                                  mode=mode, offset=np.float32().itemsize)

            instance.matrix = sp.coo_matrix((data,
                                             (rows, cols)),
                                            shape=(len(instance.dictionary),
                                                   len(instance.dictionary)),
                                            dtype=np.float32)

        return instance
