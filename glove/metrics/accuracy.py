try:
    from itertools import izip
except ImportError:
    izip = zip
import numpy as np

from .accuracy_cython import compute_rank_violations


def read_analogy_file(filename):
    """
    Read the analogy task test set from a file.
    """
    
    section = None

    with open(filename, 'r') as questions_file:
        for line in questions_file:
            if line.startswith(':'):
                section = line[2:].replace('\n', '')
                continue
            else:
                words = line.replace('\n', '').split(' ')

            yield section, words


def construct_analogy_test_set(test_examples, dictionary, ignore_missing=False):
    """
    Construct the analogy test set by mapping the words to their
    word vector ids.

    Arguments:
    - test_examples: iterable of 4-word iterables
    - dictionay: a mapping from words to ids
    - boolean ignore_missing: if True, words in the test set
                              that are not in the dictionary
                              will be dropeed.

    Returns:
    - a N by 4 numpy matrix.
    """

    test = []
    
    for example in test_examples:
        try:
            test.append([dictionary[word] for word in example])
        except KeyError:
            if ignore_missing:
                pass
            else:
                raise

    try:
        test = np.array(test, dtype=np.int32)
    except ValueError as e:
        # This should use raise ... from ... in Python 3.
        raise ValueError('Each row of the test set should contain '
                        '4 integer word ids', e)

    return test


def analogy_rank_score(analogies, word_vectors, no_threads=1):
    """
    Calculate the analogy rank score for the given set of analogies.

    A rank of zero denotes a perfect score; with random word vectors
    we would expect a rank of 0.5.

    Arguments:
    - analogies: a numpy array holding the ids of the words in the analogy tasks,
                 as constructed by `construct_analogy_test_set`.
    - word_vectors: numpy array holding the word vectors to use.
    - num_threads: number of parallel threads to use in the calculation.

    Returns:
    - ranks: a numpy array holding the normalized rank of the target word
             in each analogy task. Rank 0 means that the target words was
             returned first; rank 1 means it was returned last.
    """

    # The mean of the vectors for the
    # second, third, and the negative of
    # the first word.
    input_vectors = (word_vectors[analogies[:, 1]]
                     + word_vectors[analogies[:, 2]]
                     - word_vectors[analogies[:, 0]])

    word_vector_norms = np.linalg.norm(word_vectors,
                                       axis=1)

    # Pre-allocate the array storing the rank violations
    rank_violations = np.zeros(input_vectors.shape[0], dtype=np.int32)

    compute_rank_violations(word_vectors,
                            word_vector_norms,
                            input_vectors,
                            analogies[:, 3],
                            analogies,
                            rank_violations,
                            no_threads)

    return rank_violations / float(word_vectors.shape[0])
