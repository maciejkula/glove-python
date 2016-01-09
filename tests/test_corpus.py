# -*- coding: utf-8 -*-
import array

import pytest

import numpy as np
import scipy.sparse as sp


from glove import Corpus
from glove.glove import check_random_state


def test_corpus_construction():

    corpus_words = ['a', 'naïve', 'fox']
    corpus = [corpus_words]

    model = Corpus()
    model.fit(corpus, max_map_size=0, window=10)

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
    model.fit(corpus, max_map_size=0, window=10)

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
        model.fit(corpus, max_map_size=0, window=10)


def test_supplied_dict_missing_ignored():

    dictionary = {'a': 0,
                  'fox': 1}

    corpus = [['a', 'naïve', 'fox']]

    model = Corpus(dictionary=dictionary)
    model.fit(corpus, max_map_size=0, window=10, ignore_missing=True)

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


def _generate_training_corpus(num_sentences,
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


def _build_coocurrence_matrix(sentences):

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
                                len(dictionary))).tocsr().tocoo()


def test_large_corpus_construction():

    num_sentences = 5000
    seed = 10

    corpus = Corpus()

    corpus.fit(_generate_training_corpus(num_sentences, seed=seed))

    matrix = corpus.matrix.tocsr().tocoo()
    check_matrix = _build_coocurrence_matrix(_generate_training_corpus(num_sentences,
                                                                       seed=seed))

    assert (matrix.row == check_matrix.row).all()
    assert (matrix.col == check_matrix.col).all()
    assert np.allclose(matrix.data, check_matrix.data)
