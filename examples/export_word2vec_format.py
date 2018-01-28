from glove import Glove
import argparse

# Convert binary model to standardized .vec format for compatibility
# Example command: python export_word2vec_format.py -i model.model -o model.vec
if __name__ == '__main__':
    # Set up command line parameters.
    parser = argparse.ArgumentParser(description='Export model to word2vec format')
    parser.add_argument("-i", "--input", type=str, default=None, help="input model")
    parser.add_argument("-o", "--output", type=str, default=None, help="output model")
    args = parser.parse_args()
    glove = Glove.load(args.input)
    glove.save_word2vec_format(args.output)
