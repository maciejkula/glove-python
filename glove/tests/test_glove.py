# -*- coding: utf-8 -*-
from glove import Glove


def test_stanford_loading():

    model = Glove.load_stanford('glove/tests/stanford_test.txt')

    assert model.word_vectors is not None
    assert model.word_vectors.shape == (100, 25)
    assert len(model.dictionary) == 100

    # Python 2/3 compatibility. Check the ellipsis
    # character is in the dictionary.
    try:
        # Python 2
        assert unichr(8230) in model.dictionary
    except NameError:
        # Pyton 3
        assert '…' in model.dictionary
