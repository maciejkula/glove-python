#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#distutils: language = c++

import numpy as np
import scipy.sparse as sp


from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline int int_max(int a, int b): return a if a > b else b


cdef extern from "math.h":
    double c_abs "fabs"(double)


cdef void increment_cooc(int inner_word_key,
                         int outer_word_key,
                         double value,
                         map[pair[int, int], double]& cooc) nogil:
        """
        Increment the collocation matrix map.
        """

        cdef pair[int, int] *cooc_key

        if inner_word_key < outer_word_key:
            cooc_key = new pair[int, int](inner_word_key, outer_word_key)
        else:
            cooc_key = new pair[int, int](outer_word_key, inner_word_key)
            
        cooc[deref(cooc_key)] += value

        del cooc_key


cdef cooccurrence_map_to_matrix(int dim, map[pair[int, int], double]& cooc):
    """
    Creates a scipy.sparse.coo_matrix from the cooccurrence map.
    """

    no_collocations = cooc.size()

    # Create the constituent numpy arrays.
    row = np.empty(no_collocations, dtype=np.int32)
    col = np.empty(no_collocations, dtype=np.int32)
    data = np.empty(no_collocations, dtype=np.float64)
    cdef int[:,] row_view = row
    cdef int[:,] col_view = col
    cdef double[:,] data_view = data

    # Iteration variables
    cdef int i = 0
    cdef pair[pair[int, int], double] val
    cdef map[pair[int, int], double].iterator it = cooc.begin()


    # Iterate over the map and populate the arrays.
    while it != cooc.end():
        val = deref(it)
        
        row_view[i] = val.first.first
        col_view[i] = val.first.second
        data_view[i] = val.second

        i += 1
        inc(it)

    # Create and return the matrix.
    return sp.coo_matrix((data, (row, col)),
                         shape=(dim,
                                dim),
                         dtype=np.float64)


cdef inline int words_to_ids(list words, vector[int]& word_ids,
                      dictionary, int supplied):
    """
    Convert a list of words into a vector of word ids, using either
    the supplied dictionary or by consructing a new one.

    If the dictionary was supplied and a word is missing from it
    an error value of -1 is returned.
    """

    cdef int word_id

    word_ids.clear()

    if supplied == 1:
        for word in words:
            # Raise an error if the word
            # is missing from the supplied
            # dictionary.
            word_id = dictionary.get(word, -1)
            if word_id == -1:
                return -1
            word_ids.push_back(word_id)

    else:
        for word in words:
            word_id = dictionary.setdefault(word,
                                            len(dictionary))
            word_ids.push_back(word_id)

    return 0

            
def construct_cooccurrence_matrix(corpus, dictionary, int supplied, int window_size):
    """
    Construct the word-id dictionary and cooccurrence matrix for
    a given corpus, using a given window size.

    Returns the dictionary and a scipy.sparse COO cooccurrence matrix.
    """

    # Declare the cooccurrence map
    cdef map[pair[int, int], double] cooc

    # String processing variables.
    cdef list words
    cdef int i, j, outer_word, inner_word
    cdef int wordslen, window_stop, error
    cdef vector[int] word_ids

    # Pre-allocate some reasonable size
    # for the word ids vector.
    word_ids.reserve(1000)

    # Iterate over the corpus.
    for words in corpus:

        # Convert words to a numeric vector.
        error = words_to_ids(words, word_ids, dictionary, supplied)
        if error == -1:
            raise KeyError('Word missing from dictionary')
        wordslen = word_ids.size()

        # Record co-occurrences in a moving window.
        for i in range(wordslen):
            outer_word = word_ids[i]

            window_stop = int_min(i + window_size + 1, wordslen)

            for j in range(i, window_stop):
                inner_word = word_ids[j]

                # Do nothing if the words are the same.
                if inner_word == outer_word:
                    continue

                # Increment the matrix entry.
                increment_cooc(inner_word,
                               outer_word,
                               1.0 / (j - i),
                               cooc)
    
    # Create the matrix.
    mat = cooccurrence_map_to_matrix(len(dictionary), cooc)

    return mat
