# glove-python

A toy python implementation of [GloVe](http://www-nlp.stanford.edu/projects/glove/).

Glove produces dense vector embeddings of words, where words that occur together are close in the resulting vector space.

While this produces embeddings which are similar to [word2vec](https://code.google.com/p/word2vec/) (which has a great python implementation in [gensim](http://radimrehurek.com/gensim/models/word2vec.html)), the method is different: GloVe produces embeddings by factorizing the logarithm of the corpus word co-occurrence matrix.

The code uses asynchronous stochastic gradient descent, and is implemented in Cython. Most likely, it contains a tremendous amount of bugs.

## Installation
1. Clone this repository.
2. Make sure you have the following packages installed:
   - `numpy`
   - `scipy`
   - `cython`
   - for running the examples, `gensim`.
   - On OSX, you'll need to install `gcc` for `OpenMP` support. The setup script uses `gcc-4.9`, but you can probably change that.
3. Run `make all` to compile the cython files.

## Usage
Producing the embeddings is a two-step process: creating a co-occurrence matrix from the corpus, and then using it to produce the embeddings. The `Corpus` class helps in constructing a corpus from an interable of tokens; the `Glove` class trains the embeddings (with a sklearn-esque API).

## Examples
`example.py` has some example code for running simple training scripts: `ipython -i -- example.py -c my_corpus.txt -t 10` should process your corpus, run 10 training epochs of GloVe, and drop you into an `ipython` shell where `glove.most_similar('physics')` should produce a list of similar words.

If you want to process a wikipedia corpus, you can pass file from [here](http://dumps.wikimedia.org/enwiki/latest/) into the `example.py` script using the `-w` flag. Running `make all-wiki` should download a small wikipedia dump file, process it, and train the embeddings.

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
