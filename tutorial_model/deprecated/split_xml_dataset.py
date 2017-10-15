from xml import sax
from glob import glob
import json

DIR = 'nltk_data/corpora/stack_ex/normalized_files/'


class Splitter(sax.handler.ContentHandler):
    def __init__(self):
        super().__init__()
        self.total_file_count = 0
        self.curpath = []
        self.rows = []

    def startElement(self, name, attrs):
        if name == 'row':
            vals = list(
                filter(
                    lambda t: True if t[0] == 'Body' else False, attrs.items()
                )
            )
            if vals is None or len(vals) == 0 or vals[0][1].rstrip() == '':
                return
            self.rows.append(json.dumps({'body': vals[0][1].rstrip()}))

    def endElement(self, name):
        if len(self.rows) >= 40000:
            with open(
                '{0}file_{1}.txt'.format(DIR, self.total_file_count),
                'w'
            ) as f:
                f.write('\n'.join(self.rows))
            self.clearFields()

    def characters(self, data):
        pass

    def clearFields(self):
        self.rows = []
        self.total_file_count += 1
        print('Total File Count: {}'.format(self.total_file_count))


post_file_ids = glob('nltk_data/corpora/stack_ex/posts/*')

parser = sax.make_parser()

parser.setContentHandler(Splitter())

for f in post_file_ids:
    with open(f, 'r') as of:
        parser.parse(of)
