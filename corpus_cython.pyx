#!python
#cython: boundscheck=False, wraparound=False

import numpy as np
import scipy.sparse as sp


cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline int int_max(int a, int b): return a if a > b else b


cdef extern from "math.h":
    double c_abs "fabs"(double)


def construct_cooccurrence_matrix(corpus, dict dct, int window_size):
    """
    Construct the cooccurrence matrix for a given corpus, using
    a word-id dictionary dict and a given window size.

    Returns scipy.sparse COO cooccurrence matrix.
    """

    # Instantiate the cooccurrence matrix.
    cdef int mat_dim = len(dct)
    mat = sp.lil_matrix((mat_dim, mat_dim), dtype=np.float64)

    cdef list words
    cdef str inner_word, outer_word
    cdef int i, j, outer_word_key, inner_word_key
    cdef int wordslen, window_start, window_stop
    cdef double v

    # Low-level scipy.sparse functions without
    # type checking.
    lil_insert = sp._csparsetools.lil_insert
    lil_get1 = sp._csparsetools.lil_get1

    # Avoid attribute lookups.
    rows = mat.rows
    data = mat.data
    mat_dtype = mat.dtype

    # Iterate over the corpus.
    for words in corpus:
        wordslen = len(words)

        for i in range(wordslen):
            outer_word = words[i]

            # Define and iterate over the context window for
            # the current word.
            window_start = int_max(i - window_size, 0)
            window_stop = int_min(i + window_size, wordslen)

            for j in range(window_stop - window_start):
                inner_word = words[window_start + j]

                inner_word_key = dct[inner_word]
                outer_word_key = dct[outer_word]

                if inner_word_key == outer_word_key:
                    continue

                # Increment the matrix entry.
                v = lil_get1(mat_dim, mat_dim,
                             rows, data,
                             inner_word_key, outer_word_key)
                v += 0.5 / c_abs(i - (window_start + j))
                lil_insert(mat_dim, mat_dim,
                           rows, data,
                           inner_word_key, outer_word_key, v, mat_dtype)

    return mat.tocoo()
