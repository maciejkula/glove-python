# -*- coding: utf-8 -*-
import numpy as np

from glove import Corpus, Glove

from utils import generate_training_corpus

import pytest

def _reproduce_input_matrix(glove_model):

    wvec = glove_model.word_vectors
    wbias = glove_model.word_biases

    out = np.dot(wvec, wvec.T)

    for i in range(wvec.shape[0]):
        for j in range(wvec.shape[0]):
            if i == j:
                out[i, j] = 0.0
            elif i < j:
                out[i, j] += wbias[i] + wbias[j]
            else:
                out[i, j] = 0.0

    return np.asarray(out)


def test_stanford_loading():

    model = Glove.load_stanford('tests/stanford_test.txt')

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


def test_fitting():
    """
    Verify that the square error diminishes with fitting
    """

    num_sentences = 5000
    seed = 10

    corpus = Corpus()

    corpus.fit(generate_training_corpus(num_sentences,
                                        vocabulary_size=50,
                                        seed=seed))

    # Check that the performance is poor without fitting
    glove_model = Glove(no_components=100, learning_rate=0.05)
    glove_model.fit(corpus.matrix,
                    epochs=0,
                    no_threads=2)

    log_cooc_mat = corpus.matrix.copy()
    log_cooc_mat.data = np.log(log_cooc_mat.data)
    log_cooc_mat = np.asarray(log_cooc_mat.todense())

    repr_matrix = _reproduce_input_matrix(glove_model)

    assert ((repr_matrix - log_cooc_mat) ** 2).sum() > 30000.0

    # Check that it is good with fitting
    glove_model = Glove(no_components=100, learning_rate=0.05)
    glove_model.fit(corpus.matrix,
                    epochs=500,
                    no_threads=2)

    repr_matrix = _reproduce_input_matrix(glove_model)

    assert ((repr_matrix - log_cooc_mat) ** 2).sum() < 1500.0

def test_word_vector_by_word():
    glove_model = Glove()
    glove_model.word_vectors = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    glove_model.dictionary = {
      "first": 0,
      "second": 1
    }

    result0 = glove_model.word_vector_by_word("first")

    assert(result0 == [1, 2, 3])

    result1 = glove_model.word_vector_by_word("second")

    assert(result1 == [4, 5, 6])

def test_word_vector_by_word_without_fitting():
    glove_model = Glove()
    glove_model.dictionary = {"word": 0}

    with pytest.raises(Exception) as ex:
        glove_model.word_vector_by_word("word")

    assert(ex.value.message == "Model must be fit before querying")

def test_word_vector_by_word_without_dictionary():
    glove_model = Glove()
    glove_model.word_vectors = [[1, 2, 3]]

    with pytest.raises(Exception) as ex:
        glove_model.word_vector_by_word("word")

    assert(ex.value.message == "No word dictionary supplied")

def test_word_vector_by_word_without_word():
    glove_model = Glove()
    glove_model.word_vectors = [[1, 2, 3]]
    glove_model.dictionary = {"other": 0}

    with pytest.raises(Exception) as ex:
        glove_model.word_vector_by_word("word")

    assert(ex.value.message == "Word not in dictionary")
