import array
import timeit

import numpy as np
import scipy.sparse as sp

from glove import Corpus
from glove.glove import check_random_state


def generate_training_corpus(num_sentences,
                             vocabulary_size=30000,
                             sentence_min_size=2,
                             sentence_max_size=30,
                             seed=None):

    rs = check_random_state(seed)

    for _ in range(num_sentences):
        sentence_size = rs.randint(sentence_min_size,
                                   sentence_max_size)
        yield [str(x) for x in
               rs.randint(0, vocabulary_size, sentence_size)]


def fit_corpus(corpus):

    model = Corpus()
    model.fit(corpus)

    return corpus


if __name__ == '__main__':

    number = 10

    elapsed = timeit.timeit('fit_corpus(corpus)',
                            setup=('from __main__ import generate_training_corpus;'
                                   'from __main__ import fit_corpus;'
                                   'corpus = list(generate_training_corpus(100000, seed=10))'),
                            number=number)

    one_loop_time = elapsed / number

    print('Seconds per fit: %s' % one_loop_time)
