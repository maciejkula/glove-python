#!python
#cython: boundscheck=False, wraparound=False
#distutils: language = c++

# import numpy as np
import numpy as np
import scipy.sparse as sp


from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.string cimport string


cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline int int_max(int a, int b): return a if a > b else b


cdef extern from "math.h":
    double c_abs "fabs"(double)


cdef inline int get_word_id(string word, unordered_map[string, int]& dictionary):

    cdef int word_key
    cdef unordered_map[string,int].iterator it = dictionary.find(word)

    if it == dictionary.end():
        word_key = dictionary.size()
        dictionary.insert(pair[string,int](word, word_key))
    else:
        word_key = deref(it).second

    return word_key


cdef inline void increment_cooc(int inner_word_key,
                                int outer_word_key,
                                double value,
                                map[pair[int, int], double]& cooc):

        cdef pair[int, int] cooc_key

        if inner_word_key < outer_word_key:
            cooc_key = pair[int, int](inner_word_key, outer_word_key)
        else:
            cooc_key = pair[int, int](outer_word_key, inner_word_key)

        cooc[cooc_key] += value


def construct_cooccurrence_matrix(corpus, dict dct, int window_size):
    """
    Construct the cooccurrence matrix for a given corpus, using
    a word-id dictionary dict and a given window size.

    Returns scipy.sparse COO cooccurrence matrix.
    """

    # Declare the word dictionary and the cooccurrence map
    cdef unordered_map[string, int] dictionary
    cdef map[pair[int, int], double] cooc
    cdef int no_collocations

    cdef list words
    cdef str inner_word, outer_word
    cdef int i, j, outer_word_key, inner_word_key
    cdef int wordslen, window_start, window_stop

    # Iterate over the corpus.
    for words in corpus:
        wordslen = len(words)

        for i in range(wordslen):
            outer_word = words[i]

            # Update the mapping
            outer_word_key = get_word_id(outer_word, dictionary)

            # Define and iterate over the context window for
            # the current word.
            window_start = int_max(i - window_size, 0)
            window_stop = int_min(i + window_size, wordslen)

            for j in range(window_stop - window_start):
                inner_word = words[window_start + j]

                # inner_word_key = dct[inner_word]
                inner_word_key = get_word_id(inner_word, dictionary)

                if inner_word_key == outer_word_key:
                    continue

                # Increment the matrix entry.
                increment_cooc(inner_word_key,
                               outer_word_key,
                               0.5 / c_abs(i - (window_start + j)),
                               cooc)
    
    no_collocations = cooc.size()
    row = np.empty(no_collocations, dtype=np.int32)
    col = np.empty(no_collocations, dtype=np.int32)
    data = np.empty(no_collocations, dtype=np.float64)

    cdef int[:,] row_view = row
    cdef int[:,] col_view = col
    cdef double[:,] data_view = data

    cdef int r, c
    cdef pair[pair[int, int], double] val

    i = 0
    cdef map[pair[int, int], double].iterator it = cooc.begin()
    while it != cooc.end():
        val = deref(it)
        
        row_view[i] = val.first.first
        col_view[i] = val.first.second
        data_view[i] = val.second

        i += 1
        inc(it)


    mat = sp.coo_matrix((data, (row, col)),
                        shape=(no_collocations,
                               no_collocations),
                        dtype=np.float64)

    return dictionary, mat
