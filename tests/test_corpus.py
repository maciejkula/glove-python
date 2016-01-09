# -*- coding: utf-8 -*-
import array

import pytest

import numpy as np
import scipy.sparse as sp

from glove import Corpus
from glove.glove import check_random_state

from utils import (build_coocurrence_matrix,
                   generate_training_corpus)


def test_corpus_construction():

    corpus_words = ['a', 'naïve', 'fox']
    corpus = [corpus_words]

    model = Corpus()
    model.fit(corpus, window=10)

    for word in corpus_words:
        assert word in model.dictionary

    assert model.matrix.shape == (len(corpus_words),
                                  len(corpus_words))

    expected = [[0.0, 1.0, 0.5],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0]]

    assert (model.matrix.todense().tolist()
            == expected)


def test_supplied_dictionary():

    dictionary = {'a': 2,
                  'naïve': 1,
                  'fox': 0}

    corpus = [['a', 'naïve', 'fox']]

    model = Corpus(dictionary=dictionary)
    model.fit(corpus, window=10)

    assert model.dictionary == dictionary

    assert model.matrix.shape == (len(dictionary),
                                  len(dictionary))

    assert (model.matrix.tocsr()[2]).sum() == 0


def test_supplied_dict_checks():

    dictionary = {'a': 4,
                  'naïve': 1,
                  'fox': 0}

    with pytest.raises(Exception):
        Corpus(dictionary=dictionary)


def test_supplied_dict_missing():

    dictionary = {'a': 1,
                  'naïve': 0}

    corpus = [['a', 'naïve', 'fox']]

    model = Corpus(dictionary=dictionary)

    with pytest.raises(KeyError):
        model.fit(corpus, window=10)


def test_supplied_dict_missing_ignored():

    dictionary = {'a': 0,
                  'fox': 1}

    corpus = [['a', 'naïve', 'fox']]

    model = Corpus(dictionary=dictionary)
    model.fit(corpus, window=10, ignore_missing=True)

    assert model.dictionary == dictionary

    assert model.matrix.shape == (len(dictionary),
                                  len(dictionary))

    # Ensure that context windows and context window
    # weights are preserved.
    full_model = Corpus()
    full_model.fit(corpus, window=10)

    assert (full_model.matrix.todense()[0, 2]
            == model.matrix.todense()[0, 1]
            == 0.5)


def test_large_corpus_construction():

    num_sentences = 5000
    seed = 10

    corpus = Corpus()

    corpus.fit(generate_training_corpus(num_sentences, seed=seed))

    matrix = corpus.matrix.tocsr().tocoo()
    check_matrix = build_coocurrence_matrix(generate_training_corpus(num_sentences,
                                                                     seed=seed))

    assert (matrix.row == check_matrix.row).all()
    assert (matrix.col == check_matrix.col).all()
    assert np.allclose(matrix.data, check_matrix.data)
    assert (matrix.data > 0).all()
