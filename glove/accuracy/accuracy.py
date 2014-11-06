

def read_evaluation_file(filename):

        section = None

        with open(filename, 'rb') as questions_file:
            for line in questions_file:

                if line.startswith(':'):
                    section = line[:2]
                else:
                    words = line.split(' ')

        yield section, words

class EmbeddingAccuracy(object):

    def __init__(self, filename, dictionary=None):

        self.filename = filename
        self.dictionary = dictionary

    def read_questions(self, filename):



                

    def _is_section(self, line):

        return line.startswith(':')

    def _get_section_name(self, line):

        return line[2:]
