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


cdef class Matrix:
    """
    A sparse co-occurrence matrix storing
    its data as a vector of maps.
    """

    cdef int max_map_size
    cdef vector[unordered_map[int, float]] rows

    cdef vector[vector[int]] row_indices
    cdef vector[vector[float]] row_data

    def __cinit__(self, int max_map_size):

        self.max_map_size = max_map_size
        self.rows = vector[unordered_map[int, float]]()

        self.row_indices = vector[vector[int]]()
        self.row_data = vector[vector[float]]()

    cdef void compactify_row(self, int row):
        """
        Move a row from a map to more efficient
        vector storage.
        """

        cdef int i, col
        cdef int row_length = self.row_indices[row].size()

        cdef pair[int, float] row_entry
        cdef unordered_map[int, float].iterator row_iterator

        row_unordered_map = self.rows[row]

        # Go through the elements already in vector storage
        # and update them with the contents of a map, removing
        # map elements as they are transferred.
        for i in range(row_length):
            col = self.row_indices[row][i]
            if self.rows[row].find(col) != self.rows[row].end():

                self.row_data[row][i] += self.rows[row][col]
                self.rows[row].erase(col)

        # Resize the vectors to accommodate new
        # columns from the map.
        row_length = self.row_indices[row].size()
        self.row_indices[row].resize(row_length)
        self.row_data[row].resize(row_length)

        # Add any new columns to the vector.
        row_iterator = self.rows[row].begin()
        while row_iterator != self.rows[row].end():
            row_entry = deref(row_iterator)
            self.row_indices[row].push_back(row_entry.first)
            self.row_data[row].push_back(row_entry.second)
            inc(row_iterator)

        self.rows[row].clear()

    cdef void add_row(self):
        """
        Add a new row to the matrix.
        """

        cdef unordered_map[int, float] row_map

        row_map = unordered_map[int, float]()

        self.rows.push_back(row_map)
        self.row_indices.push_back(vector[int]())
        self.row_data.push_back(vector[float]())

    cdef void increment(self, int row, int col, float value):
        """
        Increment the value at (row, col) by value.
        """

        cdef float current_value

        while row >= self.rows.size():
            self.add_row()        

        self.rows[row][col] += value

        if self.rows[row].size() > self.max_map_size:
            self.compactify_row(row)

    cdef int size(self):
        """
        Get number of nonzero entries.
        """

        cdef int i
        cdef int size = 0

        for i in range(self.rows.size()):
            size += self.rows[i].size()
            size += self.row_indices[i].size()

        return size

    cpdef to_coo(self, int shape):
        """
        Convert to a shape by shape COO matrix.
        """

        cdef int i, j
        cdef int row
        cdef int col
        cdef int rows = self.rows.size()
        cdef int no_collocations

        # Transform all row maps to row arrays.
        for i in range(rows):
            self.compactify_row(i)

        no_collocations = self.size()

        # Create the constituent numpy arrays.
        row_np = np.empty(no_collocations, dtype=np.int32)
        col_np = np.empty(no_collocations, dtype=np.int32)
        data_np = np.empty(no_collocations, dtype=np.float64)
        cdef int[:,] row_view = row_np
        cdef int[:,] col_view = col_np
        cdef double[:,] data_view = data_np

        j = 0

        for row in range(rows):
            for i in range(self.row_indices[row].size()):

                row_view[j] = row
                col_view[j] = self.row_indices[row][i]
                data_view[j] = self.row_data[row][i]

                j += 1

        # Create and return the matrix.
        return sp.coo_matrix((data_np, (row_np, col_np)),
                             shape=(shape,
                                    shape),
                             dtype=np.float64)

    def __dealloc__(self):

        self.rows.clear()
        self.row_indices.clear()
        self.row_data.clear()


cdef inline int words_to_ids(list words, vector[int]& word_ids,
                      dictionary, int supplied, int ignore_missing):
    """
    Convert a list of words into a vector of word ids, using either
    the supplied dictionary or by consructing a new one.

    If the dictionary was supplied, a word is missing from it,
    and we are not ignoring out-of-vocabulary (OOV) words, an
    error value of -1 is returned.

    If we have an OOV word and we do want to ignore them, we use
    a -1 placeholder for it in the word_ids vector to preserve
    correct context windows (otherwise words that are far apart
    with the full vocabulary could become close together with a
    filtered vocabulary).
    """

    cdef int word_id

    word_ids.resize(0)

    if supplied == 1:
        for word in words:
            # Raise an error if the word
            # is missing from the supplied
            # dictionary.
            word_id = dictionary.get(word, -1)
            if word_id == -1 and ignore_missing == 0:
                return -1

            word_ids.push_back(word_id)

    else:
        for word in words:
            word_id = dictionary.setdefault(word,
                                            len(dictionary))
            word_ids.push_back(word_id)

    return 0

            
def construct_cooccurrence_matrix(corpus, dictionary, int supplied,
                                  int window_size, int ignore_missing,
                                  int max_map_size):
    """
    Construct the word-id dictionary and cooccurrence matrix for
    a given corpus, using a given window size.

    Returns the dictionary and a scipy.sparse COO cooccurrence matrix.
    """

    # Declare the cooccurrence map
    cdef Matrix matrix = Matrix(max_map_size)

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
        error = words_to_ids(words, word_ids, dictionary, supplied, ignore_missing)
        if error == -1:
            raise KeyError('Word missing from dictionary')
        wordslen = word_ids.size()

        # Record co-occurrences in a moving window.
        for i in range(wordslen):
            outer_word = word_ids[i]

            # Continue if we have an OOD token.
            if outer_word == -1:
                continue

            window_stop = int_min(i + window_size + 1, wordslen)

            for j in range(i, window_stop):
                inner_word = word_ids[j]

                if inner_word == -1:
                    continue

                # Do nothing if the words are the same.
                if inner_word == outer_word:
                    continue

                if inner_word < outer_word:
                    matrix.increment(inner_word,
                                     outer_word,
                                     1.0 / (j - i))
                else:
                    matrix.increment(outer_word,
                                     inner_word,
                                     1.0 / (j - i))
    
    # Create the matrix.
    mat = matrix.to_coo(len(dictionary))

    return mat
