# -*- coding: utf-8 -*-

import numpy as np
import random
import timeit
import uuid


from glove import Corpus


def test_corpus_construction():

    corpus_words = ['a', 'naïve', 'fox']
    corpus = [corpus_words]

    model = Corpus()
    model.fit(corpus, window=10)

    for word in corpus_words:
        assert word in model.dictionary

    print(model.dictionary)

    assert model.matrix.shape == (len(corpus_words),
                                  len(corpus_words))
