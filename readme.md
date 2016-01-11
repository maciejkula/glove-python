# glove-python

[![Circle CI](https://circleci.com/gh/maciejkula/glove-python.svg?style=svg)](https://circleci.com/gh/maciejkula/glove-python)

A toy python implementation of [GloVe](http://www-nlp.stanford.edu/projects/glove/).

Glove produces dense vector embeddings of words, where words that occur together are close in the resulting vector space.

While this produces embeddings which are similar to [word2vec](https://code.google.com/p/word2vec/) (which has a great python implementation in [gensim](http://radimrehurek.com/gensim/models/word2vec.html)), the method is different: GloVe produces embeddings by factorizing the logarithm of the corpus word co-occurrence matrix.

The code uses asynchronous stochastic gradient descent, and is implemented in Cython. Most likely, it contains a tremendous amount of bugs.

## Installation
Install from pypi using pip: `pip install glove_python`.

Note for OSX users: due to its use of OpenMP, glove-python does not compile under Clang. To install it, you will need a reasonably recent version of `gcc` (from Homebrew for instance). This should be picked up by `setup.py`; if it is not, please open an issue.

Building with the default Python distribution included in OSX is also not supported; please try the version from Homebrew or Anaconda.

## Usage
Producing the embeddings is a two-step process: creating a co-occurrence matrix from the corpus, and then using it to produce the embeddings. The `Corpus` class helps in constructing a corpus from an interable of tokens; the `Glove` class trains the embeddings (with a sklearn-esque API).

There is also support for rudimentary pagragraph vectors. A paragraph vector (in this case) is an embedding of a paragraph (a multi-word piece of text) in the word vector space in such a way that the paragraph representation is close to the words it contains, adjusted for the frequency of words in the corpus (in a manner similar to tf-idf weighting). These can be obtained after having trained word embeddings by calling the `transform_paragraph` method on the trained model.

## Examples
`example.py` has some example code for running simple training scripts: `ipython -i -- examples/example.py -c my_corpus.txt -t 10` should process your corpus, run 10 training epochs of GloVe, and drop you into an `ipython` shell where `glove.most_similar('physics')` should produce a list of similar words.

If you want to process a wikipedia corpus, you can pass file from [here](http://dumps.wikimedia.org/enwiki/latest/) into the `example.py` script using the `-w` flag. Running `make all-wiki` should download a small wikipedia dump file, process it, and train the embeddings. Building the cooccurrence matrix will take some time; training the vectors can be speeded up by increasing the training parallelism to match the number of physical CPU cores available.

Running this on my machine yields roughly the following results:

```
In [1]: glove.most_similar('physics')
Out[1]:
[('biology', 0.89425889335342257),
 ('chemistry', 0.88913708236100086),
 ('quantum', 0.88859617025616333),
 ('mechanics', 0.88821824562025431)]

In [4]: glove.most_similar('north')
Out[4]:
[('west', 0.99047203572917908),
 ('south', 0.98655786905501008),
 ('east', 0.97914140138065575),
 ('coast', 0.97680427897282185)]

In [6]: glove.most_similar('queen')
Out[6]:
[('anne', 0.88284931171714842),
 ('mary', 0.87615260138308615),
 ('elizabeth', 0.87362497374226267),
 ('prince', 0.87011034923161801)]

In [19]: glove.most_similar('car')
Out[19]:
[('race', 0.89549347066796814),
 ('driver', 0.89350343749207217),
 ('cars', 0.83601334715106568),
 ('racing', 0.83157724991920212)]
```

## Development
Pull requests are welcome.

When making changes to the `.pyx` extension files, you'll need to run `python setup.py cythonize` in order to produce the extension `.c` and `.cpp` files before running `pip install -e .`.
