#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Evaluation on the "analogy task", aka A is to B is C is to ?

Hacked out of gensim so we can directly compare word2vec and glove, using the same code.

Run with:
time python ./test_embed.py word2vec /data/shootout/title_tokens.txt.gz /data/embeddings/ /data/embeddings/questions-words.txt 2> /data/embeddings/word2vec.log

"""

import os
import sys
import logging

from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod

# for py2k/py3k compat
from six import iteritems, itervalues, string_types
from six.moves import xrange

import gensim
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.parsing import STOPWORDS

import glove

DIM = 200
DOC_LIMIT = None # 100000
TOKEN_LIMIT = 30000

logger = logging.getLogger("test_embed")


def fake_api(model):
    """
    There's no common API for analogy task eval across the potential model candidates.

    Fake the missing API by adding attributes to the model, to be used in analogy
    eval in accuracy().

    Modifies `model` in place.

    To be replaced by proper API later.

    """
    if isinstance(model, gensim.models.Word2Vec):
        model.init_sims()
        model.id2word = dict((v.index, w) for w, v in iteritems(model.vocab))
        model.word2id = utils.revdict(model.id2word)
        model.word_vectors = model.syn0norm
    elif isinstance(model, glove.Glove):
        model.word2id = dict((utils.to_unicode(w), id) for w, id in model.dictionary.iteritems())
        model.id2word = utils.revdict(model.word2id)


def accuracy(model, questions, most_similar, ok_words):
    """
    Compute accuracy of the model. `questions` is a filename where lines are
    4-tuples of words, split into sections by ": SECTION NAME" lines.
    See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

    The accuracy is reported (=printed to log and returned as a list) for each
    section separately, plus there's one aggregate summary at the end.

    Only evaluate on the set of `ok_words` (such as 30k most common words), ignoring
    any test examples where any of the four words falls outside `ok_words`.

    This method corresponds to the `compute-accuracy` script of the original C word2vec.

    """
    def log_accuracy(section):
        correct, incorrect = section['correct'], section['incorrect']
        if correct + incorrect > 0:
            logging.info("%s: %.1f%% (%i/%i)" %
                (section['section'], 100.0 * correct / (correct + incorrect),
                correct, correct + incorrect))

    sections, section = [], None
    for line_no, line in enumerate(utils.smart_open(questions)):
        # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
        line = utils.to_unicode(line)
        if line.startswith(': '):
            # a new section starts => store the old section
            if section:
                sections.append(section)
                log_accuracy(section)
            section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
        else:
            if not section:
                raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
            try:
                a, b, c, expected = [word.lower() for word in line.split()]  # TODO assumes vocabulary preprocessing uses lowercase, too...
            except:
                logging.info("skipping invalid line #%i in %s" % (line_no, questions))
            if a not in ok_words or b not in ok_words or c not in ok_words or expected not in ok_words:
                logging.debug("skipping line #%i with OOV words: %s" % (line_no, line.strip()))
                continue

            ignore = set(model.word2id[v] for v in [a, b, c])  # indexes of words to ignore
            predicted = None

            # find the most likely prediction, ignoring OOV words and input words
            for index in argsort(most_similar(model, positive=[b, c], negative=[a], topn=False))[::-1]:
                if model.id2word[index] in ok_words and index not in ignore:
                    predicted = model.id2word[index]
                    if predicted != expected:
                        logging.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                    break

            section['correct' if predicted == expected else 'incorrect'] += 1
    if section:
        # store the last section, too
        sections.append(section)
        log_accuracy(section)

    total = {'section': 'total', 'correct': sum(s['correct'] for s in sections), 'incorrect': sum(s['incorrect'] for s in sections)}
    log_accuracy(total)
    sections.append(total)
    return sections


def most_similar(model, positive=[], negative=[], topn=10):
    """
    Find the top-N most similar words. Positive words contribute positively towards the
    similarity, negative words negatively.

    This method computes cosine similarity between a simple mean of the projection
    weight vectors of the given words, and corresponds to the `word-analogy` and
    `distance` scripts in the original word2vec implementation.

    Example::

      >>> most_similar(model, positive=['woman', 'king'], negative=['man'])
      [('queen', 0.50882536), ...]

    """
    if isinstance(positive, string_types) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                            else word for word in positive]
    negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                             else word for word in negative]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, ndarray):
            mean.append(weight * word)
        elif word in model.word2id:
            word_index = model.word2id[word]
            mean.append(weight * model.word_vectors[word_index])
            all_words.add(word_index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    dists = dot(model.word_vectors, mean)
    if not topn:
        return dists
    best = argsort(dists)[::-1][:topn + len(all_words)]

    # ignore (don't return) words from the input
    result = [(model.id2word[sim], float(dists[sim])) for sim in best if sim not in all_words]

    return result[:topn]



# if __name__ == "__main__":
#     logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
#     logging.info("running %s" % " ".join(sys.argv))

#     # check and process cmdline input
#     program = os.path.basename(sys.argv[0])
#     if len(sys.argv) < 5:
#         print(globals()['__doc__'] % locals())
#         sys.exit(1)

#     method, infile, outdir, testfile = sys.argv[1:5]
#     texts = gensim.models.word2vec.LineSentence(infile)
#     texts = gensim.utils.ClippedCorpus(texts, DOC_LIMIT)
#     TODO filter texts for stopwords / replace by placeholders
#     TODO ok_words = filter_extremes down to 30k
#     TODO filter texts to ok_words / replace by OOV placeholders

#     if method == 'lsi':
#         model = gensim.models.LsiModel(corpus, id2word=corpus.dictionary, num_topics=DIM)
#         TODO
#     elif method == 'lda':
#         model = gensim.models.LdaMulticore(corpus, id2word=corpus.dictionary, num_topics=DIM, workers=8)
#         topics = model.state.get_lambda()
#         TODO
#     elif method == 'word2vec':
#         model = gensim.models.Word2Vec(texts, size=DIM, workers=7, min_count=0)
#     elif method == 'glove':
#         # glove doesn't support unicode, transform to bytestrings
#         texts_utf8 = ([utils.to_utf8(w) for w in text] for text in texts)

#         corpus_model = glove.Corpus()
#         corpus_model.fit(texts_utf8, window=10)

#         glove = Glove(no_components=DIM, learning_rate=0.05)
#         glove.fit(corpus_model.matrix, epochs=15, no_threads=8, verbose=True)
#         glove.add_dictionary(corpus_model.dictionary)

#     model.save(os.path.join(outdir, '%s_%s.pkl' % (method, DIM)))

#     accuracy(model, testfile, most_similar, ok_words)

#     logging.info("finished running %s" % program)

