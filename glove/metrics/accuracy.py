import numpy as np



def read_analogy_file(filename):
    """
    Read the analogy task test set from a file.
    """
    
    section = None

    with open(filename, 'r') as questions_file:
        for line in questions_file:
            if line.startswith(':'):
                section = line[:2]
                continue
            else:
                words = line.replace('\n', '').split(' ')

            yield section, words


def construct_analogy_test_set(test_examples, dictionary, ignore_missing=False):
    """
    Construct the analogy test set by mapping the words to their
    word vector ids.

    Arguments:
    - test_examples: iterable of 4-word iterables
    - dictionay: a mapping from words to ids
    - boolean ignore_missing: if True, words in the test set
                              that are not in the dictionary
                              will be dropeed.

    Returns:
    - a N by 4 numpy matrix.
    """

    test = []
    
    for example in test_examples:
        try:
            test.append([dictionary[word] for word in example])
        except KeyError:
            if ignore_missing:
                pass
            else:
                raise

    try:
        test = np.array(test, dtype=np.int32)
    except ValueError as e:
        # This should use raise ... from ... in Python 3.
        raise ValueError('Each row of the test set should contain '
                        '4 integer word ids', e)

    return test
