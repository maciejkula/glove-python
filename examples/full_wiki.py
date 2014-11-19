import numpy as np
import pprint
import argparse
import gensim
import itertools

from glove import Glove
from glove import Corpus



def read_wikipedia_corpus(filename):

    # We don't want to do a dictionary construction pass.
    corpus = gensim.corpora.WikiCorpus(filename, dictionary={})

    for text in corpus.get_texts():
        yield text


if __name__ == '__main__':
    
    # Set up command line parameters.
    parser = argparse.ArgumentParser(description='Fit a GloVe model.')

    parser.add_argument('--create', '-c', action='store',
                        default=None,
                        help=('The filename of the corpus to pre-process. '
                              'The pre-processed corpus will be saved '
                              'and will be ready for training.'))
    parser.add_argument('-wiki', '-w', action='store_true',
                        default=False,
                        help=('Assume the corpus input file is in the '
                              'Wikipedia dump format'))
    parser.add_argument('--train', '-t', action='store',
                        default=0,
                        help=('Train the GloVe model with this number of epochs.'
                              'If not supplied, '
                              'We\'ll attempt to load a trained model'))
    parser.add_argument('--word2vec', '-v', action='store_true',
                        default=False,
                        help=('Train the word2vec model'))
    parser.add_argument('--nan', '-n', action='store_true',
                        default=False,
                        help='Repeat 1 epoch until NaNs are found.')
    parser.add_argument('--dictionary', '-d', action='store_true',
                        default=False,
                        help='Build a dictionary; load it otherwise.')
    parser.add_argument('--parallelism', '-p', action='store',
                        default=1,
                        help=('Number of parallel threads to use for training'))
    parser.add_argument('--query', '-q', action='store',
                        default='',
                        help='Get closes words to this word.')
    args = parser.parse_args()

    if args.dictionary:
        # Build dictionary
        dct = gensim.corpora.dictionary.Dictionary(get_data(args.create))
        dct.filter_extremes(no_below=5, no_above=0.5, keep_n=30000)
        dct.save('dct.model')

    if args.create and not args.word2vec:
        # Build the corpus dictionary and the cooccurrence matrix.
        print('Pre-processing corpus')


        dct = gensim.corpora.dictionary.Dictionary.load('dct.model')

        print('Dict size: %s' % len(dct))

        corpus_model = Corpus(dictionary=dct.token2id)
        corpus_model.fit(read_wikipedia_corpus(args.create), window=10, ignore_missing=True)
        corpus_model.save('corpus.model')
        
        print('Dict size: %s' % len(corpus_model.dictionary))
        print('Collocations: %s' % corpus_model.matrix.nnz)

    if args.word2vec:

        dct = gensim.corpora.dictionary.Dictionary.load('dct.model')

        data = ([y for y in x if y in dct.token2id] for x in read_wikipedia_corpus(args.create))

        print('Building word2vec vocabulary')
        model = gensim.models.Word2Vec(workers=3, min_count=0)
        model.build_vocab(data)

        data = ([y for y in x if y in dct.token2id] for x in read_wikipedia_corpus(args.create))
        print('Training word2vec')
        model.train(data)

        model.save('word2vec.model')

    if args.train:
        # Train the GloVe model and save it to disk.

        if not args.create:
            # Try to load a corpus from disk.
            print('Reading corpus statistics')
            corpus_model = Corpus.load('corpus.model')

            print('Dict size: %s' % len(corpus_model.dictionary))
            print('Collocations: %s' % corpus_model.matrix.nnz)

        print('Training the GloVe model')

        reps = 30 if args.nan else 1

        for rep in range(reps):
            print('Rep %s' % rep)
            glove = Glove(no_components=200, learning_rate=0.01)
            glove.fit(corpus_model.matrix, epochs=int(args.train),
                      no_threads=args.parallelism, verbose=True)
            glove.add_dictionary(corpus_model.dictionary)

            if np.isnan(glove.word_vectors).sum() > 0:
                raise Exception('NaNs in word vectors')

        glove.save('glove.model')

    if args.query:
        # Finally, query the model for most similar words.
        if not args.train:
            print('Loading pre-trained GloVe model')
            glove = Glove.load('glove.model')

        print('Querying for %s' % args.query)
        pprint.pprint(glove.most_similar(args.query, number=10))
