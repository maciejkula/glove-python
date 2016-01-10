#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

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


def fit_vectors(double[:, ::1] wordvec,
                double[:, ::1] wordvec_sum_gradients,
                double[::1] wordbias,
                double[::1] wordbias_sum_gradients,
                int[::1] row,
                int[::1] col,
                double[::1] counts,
                int[::1] shuffle_indices,
                double initial_learning_rate,
                double max_count,
                double alpha,
                double max_loss,
                int no_threads):
    """
    Estimate GloVe word embeddings given the cooccurrence matrix.
    Modifies the word vector and word bias array in-place.

    Training is performed via asynchronous stochastic gradient descent,
    using the AdaGrad per-coordinate learning rate.
    """

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]

    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_a, word_b
    cdef double count, learning_rate, gradient

    # Loss and gradient variables.
    cdef double prediction, entry_weight, loss

    # Iteration variables
    cdef int i, j, shuffle_index

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    with nogil:
        for j in prange(no_cooccurrences, num_threads=no_threads,
                        schedule='dynamic'):
            shuffle_index = shuffle_indices[j]
            word_a = row[shuffle_index]
            word_b = col[shuffle_index]
            count = counts[shuffle_index]

            # Get prediction
            prediction = 0.0

            for i in range(dim):
                prediction = prediction + wordvec[word_a, i] * wordvec[word_b, i]

            prediction = prediction + wordbias[word_a] + wordbias[word_b]

            # Compute loss and the example weight.
            entry_weight = double_min(1.0, (count / max_count)) ** alpha
            loss = entry_weight * (prediction - c_log(count))

            # Clip the loss for numerical stability.
            if loss < -max_loss:
                loss = -max_loss
            elif loss > max_loss:
                loss = max_loss

            # Update step: apply gradients and reproject
            # onto the unit sphere.
            for i in range(dim):

                learning_rate = initial_learning_rate / sqrt(wordvec_sum_gradients[word_a, i])
                gradient = loss * wordvec[word_b, i]
                wordvec[word_a, i] = (wordvec[word_a, i] - learning_rate 
                                      * gradient)
                wordvec_sum_gradients[word_a, i] += gradient ** 2

                learning_rate = initial_learning_rate / sqrt(wordvec_sum_gradients[word_b, i])
                gradient = loss * wordvec[word_a, i]
                wordvec[word_b, i] = (wordvec[word_b, i] - learning_rate
                                      * gradient)
                wordvec_sum_gradients[word_b, i] += gradient ** 2

            # Update word biases.
            learning_rate = initial_learning_rate / sqrt(wordbias_sum_gradients[word_a])
            wordbias[word_a] -= learning_rate * loss
            wordbias_sum_gradients[word_a] += loss ** 2

            learning_rate = initial_learning_rate / sqrt(wordbias_sum_gradients[word_b])
            wordbias[word_b] -= learning_rate * loss
            wordbias_sum_gradients[word_b] += loss ** 2


def transform_paragraph(double[:, ::1] wordvec,
                        double[::1] wordbias,
                        double[::1] paragraphvec,
                        double[::1] sum_gradients,
                        int[::1] row,
                        double[::1] counts,
                        int[::1] shuffle_indices,
                        double initial_learning_rate,
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

    # Loss and gradient variables.
    cdef double prediction
    cdef double entry_weight
    cdef double loss
    cdef double gradient

    # Iteration variables
    cdef int epoch, i, j, shuffle_index

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    for epoch in range(epochs):
        for j in range(no_cooccurrences):
            shuffle_index = shuffle_indices[j]

            word_b = row[shuffle_index]
            count = counts[shuffle_index]

            # Get prediction
            prediction = 0.0
            for i in range(dim):
                prediction = prediction + paragraphvec[i] * wordvec[word_b, i]
            prediction += wordbias[word_b]

            # Compute loss and the example weight.
            entry_weight = double_min(1.0, (count / max_count)) ** alpha
            loss = entry_weight * (prediction - c_log(count))

            # Update step: apply gradients.
            for i in range(dim):
                learning_rate = initial_learning_rate / sqrt(sum_gradients[i])
                gradient = loss * wordvec[word_b, i]
                paragraphvec[i] = (paragraphvec[i] - learning_rate
                                   * gradient)
                sum_gradients[i] += gradient ** 2
