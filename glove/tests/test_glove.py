# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp

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


def test_shuffling():

    shape = 10

    row = np.random.randint(shape, size=shape).astype(np.int32)
    col = np.random.randint(shape, size=shape).astype(np.int32)
    data = np.random.rand(shape)

    row_copy = row.copy()
    col_copy = col.copy()
    data_copy = data.copy()

    mat = sp.coo_matrix((data, (row, col)),
                        shape=(shape, shape))
    test_mat = sp.coo_matrix((data_copy, (row_copy, col_copy)),
                             shape=(shape, shape))

    # The same before...
    assert np.all(mat.todense() == test_mat.todense())
    
    model = Glove(random_seed=10)
    model._shuffle_coo_matrix(row, col, data)

    # ...and after shuffling
    assert np.all(mat.todense() == test_mat.todense())
    assert not np.all(row == row_copy)
    assert not np.all(col == col_copy)
    assert not np.all(data == data_copy)



    
