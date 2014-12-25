#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#distutils: language = c++

import numpy as np


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


cdef class COOMatrix:
    """
    """

    cdef public rows_arr
    cdef public cols_arr
    cdef public data_arr

    cdef int memmap
    cdef row_fname
    cdef col_fname
    cdef data_fname

    cdef int[::1] rows
    cdef int[::1] cols
    cdef float[::1] data

    def __cinit__(self, row_fname, col_fname, data_fname):

        self.rows_arr = None
        self.cols_arr = None
        self.data_arr = None

        self.row_fname = row_fname
        self.col_fname = col_fname
        self.data_fname = data_fname

        if self.row_fname is not None:
            self.memmap = 1

        if self.memmap == 1:
            self.create_memmap_arrays(0, 'w+')
        else:
            self.rows_arr = np.array([], dtype=np.int32)
            self.cols_arr = np.array([], dtype=np.int32)
            self.data_arr = np.array([], dtype=np.float32)
        
        self.rows = self.rows_arr
        self.cols = self.cols_arr
        self.data = self.data_arr

    cdef void create_memmap_arrays(self, int size, str mode):
        """
        """

        self.rows_arr = np.memmap(self.row_fname, dtype=np.int32,
                                  mode=mode, offset=np.int32().itemsize, shape=size)
        self.cols_arr = np.memmap(self.col_fname, dtype=np.int32,
                                  mode=mode, offset=np.int32().itemsize, shape=size)
        self.data_arr = np.memmap(self.data_fname, dtype=np.float32,
                                  mode=mode, offset=np.float32().itemsize, shape=size)

    cdef int size(self):
        """
        """

        return self.rows.shape[0]

    cdef void resize(self, int size):
        """
        """

        mode = 'r+'

        if self.memmap == 1:
            self.create_memmap_arrays(size, 'r+')
        else:
            self.rows_arr.resize(size, refcheck=False)
            self.cols_arr.resize(size, refcheck=False)
            self.data_arr.resize(size, refcheck=False)

        self.rows = self.rows_arr
        self.cols = self.cols_arr
        self.data = self.data_arr

    cdef void set_entry(self, int pos, int row, int col, float datum):
        """
        """

        self.rows[pos] = row
        self.cols[pos] = col
        self.data[pos] = datum

    cdef void flush(self):
        """
        """

        if self.memmap == 1:
            self.rows_arr.flush()
            self.cols_arr.flush()
            self.data_arr.flush()


cdef class Matrix:
    """
    A sparse co-occurrence matrix storing
    its data as a vector of maps.
    """

    cdef int max_map_size
    cdef int map_size
    cdef COOMatrix matrix
    cdef vector[unordered_map[int, float]] rows

    def __cinit__(self, COOMatrix matrix, int max_map_size):

        self.max_map_size = max_map_size
        self.map_size = 0
        self.rows = vector[unordered_map[int, float]]()

        self.matrix = matrix

    cdef void compactify(self):
        """
        Move all entries from row maps to more efficient
        array storage.
        """

        cdef int i, row, col, new_entry_offset
        cdef pair[int, float] row_entry
        cdef unordered_map[int, float].iterator row_iterator
        cdef COOMatrix coo = self.matrix

        # print 'trying to compactify at size %s' % self.map_size

        # Increment entries already in array storage.
        for i in range(coo.size()):

            row = coo.rows[i]
            col = coo.cols[i]

            # If we have the entry in the map, move
            # it to array storage and erase from the
            # map.
            if (self.rows[row].find(col)
                != self.rows[row].end()):

                coo.data[i] += self.rows[row][col]
                self.rows[row].erase(col)
                self.map_size -= 1

        # print 'incremented existing entries'

        # All remaining entries in the map are not
        # already in the arrays: we need to resize
        # them to accommodate new entries.
        new_entry_offset = coo.size()
        coo.resize(new_entry_offset + self.map_size)

        for row in range(self.rows.size()):
            row_iterator = self.rows[row].begin()
            while row_iterator != self.rows[row].end():
                row_entry = deref(row_iterator)
                coo.set_entry(new_entry_offset,
                              row, row_entry.first,
                              row_entry.second)
                new_entry_offset += 1
                self.map_size -= 1
                inc(row_iterator)

            self.rows[row].clear()

        coo.flush()

    cdef void add_row(self):
        """
        Add a new row to the matrix map.
        """

        cdef unordered_map[int, float] row_map

        row_map = unordered_map[int, float]()
        self.rows.push_back(row_map)

        # print 'added row'

    cdef void increment(self, int row, int col, float value):
        """
        Increment the value at (row, col) by value.
        """

        cdef float current_value
        cdef int preinsert_size

        while row >= self.rows.size():
            self.add_row()

        preinsert_size = self.rows[row].size()

        self.rows[row][col] += value

        if self.rows[row].size() > preinsert_size:
            self.map_size += 1

        if self.map_size > self.max_map_size:
            self.compactify()

        # print 'Incremented'

    cdef int size(self):
        """
        Get number of nonzero entries.
        """

        cdef int i
        cdef int size = 0

        for i in range(self.rows.size()):
            size += self.rows[i].size()

        return size

    ## cpdef to_coo(self, int shape):
    ##     """
    ##     Convert to a shape by shape COO matrix.
    ##     """

    ##     self.compactify()

    ##     # Create and return the matrix.
    ##     ## mat = sp.coo_matrix((shape, shape), np.float64)
    ##     ## mat.data = self.matrix.data
    ##     ## mat.row = self.matrix.rows
    ##     ## mat.col = self.matrix.cols

    ##     ## return mat
    ##     return sp.coo_matrix((self.matrix.data,
    ##                          (self.matrix.rows, self.matrix.cols)),
    ##                          shape=(shape,
    ##                                 shape),
    ##                          dtype=np.float64)

    def __dealloc__(self):

        self.rows.clear()


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

            
def construct_cooccurrence_matrix(corpus, dictionary, COOMatrix coo_matrix,
                                  int supplied, int window_size,
                                  int ignore_missing, int max_map_size):
    """
    Construct the word-id dictionary and cooccurrence matrix for
    a given corpus, using a given window size.

    Returns the dictionary and a scipy.sparse COO cooccurrence matrix.
    """

    # print 'trying to create the matrix'
    # Declare the cooccurrence map
    cdef Matrix matrix = Matrix(coo_matrix, max_map_size)

    # print 'created matrix'

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

    matrix.compactify()
