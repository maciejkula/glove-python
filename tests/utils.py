# -*- coding: utf-8 -*-
import array

import numpy as np
import scipy.sparse as sp

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


def build_coocurrence_matrix(sentences):

    dictionary = {}
    rows = []
    cols = []
    data = array.array('f')

    window = 10

    for sentence in sentences:
        for i, first_word in enumerate(sentence):
            first_word_idx = dictionary.setdefault(first_word,
                                                   len(dictionary))
            for j, second_word in enumerate(sentence[i:i + window + 1]):
                second_word_idx = dictionary.setdefault(second_word,
                                                        len(dictionary))

                distance = j

                if first_word_idx == second_word_idx:
                    pass
                elif first_word_idx < second_word_idx:
                    rows.append(first_word_idx)

                    cols.append(second_word_idx)
                    data.append(np.float32(1.0) / distance)
                else:
                    rows.append(second_word_idx)
                    cols.append(first_word_idx)
                    data.append(np.float32(1.0) / distance)

    return sp.coo_matrix((data, (rows, cols)),
                         shape=(len(dictionary),
                                len(dictionary)),
                         dtype=np.float32).tocsr().tocoo()
