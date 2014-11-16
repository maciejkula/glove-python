import argparse
from collections import defaultdict
import numpy as np

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
                        default=False,
                        help=('If True, words from the '
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
        encode = lambda words: [unicode(x.lower()) for x in words]


    # Load the analogy task dataset. One example can be obtained at
    # https://word2vec.googlecode.com/svn/trunk/questions-words.txt
    sections = defaultdict(list)
    evaluation_words = [sections[section].append(encode(words)) for section, words in
                        metrics.read_analogy_file(args.test)]

    section_ranks = []

    for section, words in sections.items():
        evaluation_ids = metrics.construct_analogy_test_set(words,
                                                            glove.dictionary,
                                                            ignore_missing=True)

        # Get the rank array.
        ranks = metrics.analogy_rank_score(evaluation_ids, glove.word_vectors,
                                           no_threads=int(args.parallelism))
        section_ranks.append(ranks)

        print('Section %s mean rank: %s, accuracy: %s' % (section, ranks.mean(), 
                                                          (ranks == 0).sum() / float(len(ranks))))
    
    ranks = np.hstack(section_ranks)

    print('Overall rank: %s, accuracy: %s' % (ranks.mean(), 
                                              (ranks == 0).sum() / float(len(ranks))))

