#!python
#cython: boundscheck=False, wraparound=False

import numpy as np
import scipy.sparse as sp
import collections
from cython.parallel import parallel, prange


cdef inline double double_min(double a, double b) nogil: return a if a <= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b
cdef inline int int_max(int a, int b) nogil: return a if a > b else b


cdef extern from "math.h" nogil:
    double sqrt(double)
    double c_log "log"(double)
    double c_abs "fabs"(double)


def fit_vectors(double[:, :] wordvec,
                double[:,] wordbias,
                int[:,] row,
                int[:,] col,
                double[:,] counts,
                int[:,] shuffle_indices,
                double learning_rate,
                double max_count,
                double alpha,
                int no_threads):
    """
    Estimate GloVe word embeddings given the cooccurrence matrix.
    Modifies the word vector and word bias array in-place.

    Training is performed via asynchronous stochastic gradient descent.
    """

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]
    
    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_a
    cdef int word_b
    cdef double count

    # Hold norms of the word vectors.
    cdef double word_a_norm
    cdef double word_b_norm

    # Loss and gradient variables.
    cdef double prediction
    cdef double entry_weight = 0.0
    cdef double loss = 0.0

    # Iteration variables
    cdef int j, i, shuffle_index

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    with nogil:
        for j in prange(no_cooccurrences, num_threads=no_threads,
                        schedule='dynamic'):
            shuffle_index = shuffle_indices[j]
            word_a = row[shuffle_index]
            word_b = col[shuffle_index]
            count = counts[shuffle_index]

            # Get prediction, and accumulate
            # vector norms as we go.
            prediction = 0.0
            word_a_norm = 0.0
            word_b_norm = 0.0

            for i in range(dim):
                prediction = prediction + wordvec[word_a, i] * wordvec[word_b, i]
                word_a_norm += wordvec[word_a, i] ** 2
                word_b_norm += wordvec[word_b, i] ** 2

            prediction = prediction + wordbias[word_a] + wordbias[word_b]

            word_a_norm = sqrt(word_a_norm)
            word_b_norm = sqrt(word_b_norm)

            # Compute loss and the example weight.
            entry_weight = double_min(1.0, (count / max_count)) ** alpha
            loss = entry_weight * (prediction - c_log(count))

            # Update step: apply gradients and reproject
            # onto the unit sphere.
            for i in xrange(dim):
                wordvec[word_a, i] = (wordvec[word_a, i] - learning_rate 
                                      * loss * wordvec[word_b, i]) / word_a_norm
                wordvec[word_b, i] = (wordvec[word_b, i] - learning_rate
                                      * loss * wordvec[word_a, i]) / word_b_norm

            # Update word biases.
            wordbias[word_a] -= learning_rate * loss
            wordbias[word_b] -= learning_rate * loss


def transform_paragraph(double[:, :] wordvec,
                        double[:,] wordbias,
                        double[:,] paragraphvec,
                        int[:,] row,
                        double[:,] counts,
                        int[:,] shuffle_indices,
                        double learning_rate,
                        double max_count,
                        double alpha,
                        int epochs):
    """
    Compute a vector representation of a paragraph. This has
    the effect of making the paragraph vector close to words
    that occur in it. The representation should be more
    similar to words that occur in it multiple times, and 
    less close to words that are common in the corpus (have
    large word bias values).

    This should be be similar to a tf-idf weighting.
    """

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]
    
    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_b, word_a
    cdef double count

    # Hold norm of the paragraph vector.
    cdef double paragraphnorm

    # Loss and gradient variables.
    cdef double prediction
    cdef double entry_weight = 0.0
    cdef double loss = 0.0

    # Iteration variables
    cdef int epoch, j, c, i, shuffle_index, start, stop

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    for epoch in xrange(epochs):
        for j in xrange(no_cooccurrences):
            shuffle_index = shuffle_indices[j]

            word_b = row[shuffle_index]
            count = counts[shuffle_index]

            # Get prediction, and accumulate
            # vector norms as we go.
            prediction = 0.0
            paragraphnorm = 0.0

            for i in range(dim):
                prediction = prediction + paragraphvec[i] * wordvec[word_b, i]
                paragraphnorm += paragraphvec[i] ** 2

            prediction += wordbias[word_b]
            paragraphnorm = sqrt(paragraphnorm)

            # Compute loss and the example weight.
            entry_weight = double_min(1.0, (count / max_count)) ** alpha
            loss = entry_weight * (prediction - c_log(count))

            # Update step: apply gradients and reproject
            # onto the unit sphere.
            for i in xrange(dim):
                paragraphvec[i] = (paragraphvec[i] - learning_rate 
                                      * loss * wordvec[word_b, i]) / paragraphnorm
