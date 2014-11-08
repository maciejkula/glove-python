import argparse


from glove import Glove, metrics


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=('Evaluate a trained GloVe '
                                                  'model on an analogy task.'))
    parser.add_argument('--test', '-t', action='store',
                        required=True,
                        help='The filename of the analogy test set.')
    parser.add_argument('--model', '-m', action='store',
                        required=True,
                        help='The filename of the stored GloVe model.')
    parser.add_argument('--encode', '-e', action='store_true',
                        default=True,
                        help=('If True (default), words from the '
                              'evaluation set will be utf-8 encoded '
                              'before looking them up in the '
                              'model dictionary'))
    parser.add_argument('--parallelism', '-p', action='store',
                        default=1,
                        help=('Number of parallel threads to use'))

    args = parser.parse_args()

    # Load the GloVe model
    glove = Glove.load(args.model)


    if args.encode:
        encode = lambda words: [x.lower().encode('utf-8') for x in words]
    else:
        encode = lambda words: [x.lower() for x in words]


    # Load the analogy task dataset. One example can be obtained at
    # https://word2vec.googlecode.com/svn/trunk/questions-phrases.txt
    evaluation_words = [encode(words) for section, words in
                        metrics.read_analogy_file(args.test)]
    evaluation_ids = metrics.construct_analogy_test_set(evaluation_words,
                                                        glove.dictionary,
                                                        ignore_missing=True)

    # Get the rank array.
    ranks = metrics.analogy_rank_score(evaluation_ids[:1000], glove.word_vectors,
                                       no_threads=int(args.parallelism))

    print('Mean rank: %s' % ranks.mean())
    


