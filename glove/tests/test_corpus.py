# -*- coding: utf-8 -*-
from nose.tools import raises
import numpy as np
import os


from glove import Corpus


MEMMAP_PREFIX = 'test_memmap'
SAVE_FNAME = 'test_save_corpus'

def _check_memmap(matrix):

    assert not matrix.row.flags['OWNDATA']
    assert not matrix.col.flags['OWNDATA']
    assert not matrix.data.flags['OWNDATA']
    
    assert isinstance(matrix.row.base, np.memmap)
    assert isinstance(matrix.col.base, np.memmap)
    assert isinstance(matrix.data.base, np.memmap)


def test_corpus_construction():

    corpus_words = ['a', 'naïve', 'fox']
    corpus = [corpus_words]

    for memmap_prefix in (None, MEMMAP_PREFIX):
        model = Corpus(memmap_prefix=MEMMAP_PREFIX)
        model.fit(corpus, max_map_size=0, window=10)

        for word in corpus_words:
            assert word in model.dictionary

        assert model.matrix.shape == (len(corpus_words),
                                      len(corpus_words))

        expected = [[0.0, 1.0, 0.5],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]]

        assert model.matrix.data.min() > 0
        assert (model.matrix.todense().tolist()
                == expected)

        if memmap_prefix:
            _check_memmap(model.matrix)

def test_save_load():

    corpus_words = ['a', 'naïve', 'fox']
    corpus = [corpus_words]

    expected = [[0.0, 1.0, 0.5],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0]]

    for memmap_prefix in (None, MEMMAP_PREFIX):
        model = Corpus(memmap_prefix=MEMMAP_PREFIX)
        model.fit(corpus, max_map_size=0, window=10)

        model.save(SAVE_FNAME)
        model = Corpus.load(SAVE_FNAME)

        assert model.matrix.data.min() > 0
        assert (model.matrix.todense().tolist()
                == expected)

        if memmap_prefix:
            _check_memmap(model.matrix)


def test_supplied_dictionary():

    dictionary = {'a': 2,
                  'naïve': 1,
                  'fox': 0}

    corpus = [['a', 'naïve', 'fox']]

    for memmap_prefix in (None, MEMMAP_PREFIX):
        model = Corpus(dictionary=dictionary,
                       memmap_prefix=MEMMAP_PREFIX)
        model.fit(corpus, max_map_size=0, window=10)

        assert model.dictionary == dictionary

        assert model.matrix.shape == (len(dictionary),
                                      len(dictionary))

        assert (model.matrix.tocsr()[2]).sum() == 0


@raises(Exception)
def test_supplied_dict_checks():

    dictionary = {'a': 4,
                  'naïve': 1,
                  'fox': 0}

    corpus = [['a', 'naïve', 'fox']]

    model = Corpus(dictionary=dictionary)


@raises(KeyError)
def test_supplied_dict_missing():

    dictionary = {'a': 1,
                  'naïve': 0}

    corpus = [['a', 'naïve', 'fox']]

    model = Corpus(dictionary=dictionary)
    model.fit(corpus, max_map_size=0, window=10)


def test_supplied_dict_missing_ignored():

    dictionary = {'a': 0,
                  'fox': 1}

    corpus = [['a', 'naïve', 'fox']]

    for memmap_prefix in (None, MEMMAP_PREFIX):
        model = Corpus(dictionary=dictionary,
                       memmap_prefix=MEMMAP_PREFIX)
        model.fit(corpus, max_map_size=0, window=10,
                  ignore_missing=True)

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
