#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from cython.parallel import prange


cdef double dot(double[::1] x,
                double[::1] y,
                int dim) nogil:

    cdef int i
    cdef double result = 0.0

    for i in range(dim):
        result += x[i] * y[i]

    return result


def compute_rank(double[:, ::1] wordvec,
                 double[::1] wordvec_norm,
                 double[:, ::1] input,
                 int[:] expected,
                 double[::1] ranks,
                 int no_threads):
    """
    Compute the normalized rank scores (0.0 == best, 1.0 == worst)
    of the expected words in the word analogy task.
    """

    cdef int i, j, no_input_vectors, no_wordvec
    cdef int no_components

    cdef double score_of_expected, score
    cdef double rank_violations

    no_input_vectors = input.shape[0]
    no_wordvec = wordvec.shape[0]
    no_components = wordvec.shape[1]

    with nogil:
        for i in prange(no_input_vectors, num_threads=no_threads,
                        schedule='dynamic'):

            # Compute the score of the expected word.
            score_of_expected = (dot(input[i],
                                    wordvec[expected[i]],
                                    no_components)
                                 / wordvec_norm[expected[i]])

            # Compute all other scores and count
            # rank violations.
            rank_violations = 0.0

            for j in range(no_wordvec):

                if i == j:
                    continue

                score = (dot(input[i],
                            wordvec[j],
                            no_components)
                         / wordvec_norm[j])

                if score >= score_of_expected:
                    rank_violations = rank_violations + 1

            # Update the average rank with the rank
            # of this example.
            ranks[i] = rank_violations / no_wordvec
