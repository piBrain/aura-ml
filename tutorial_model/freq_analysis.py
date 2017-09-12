import nltk
import math
from glob import glob
from pprint import pprint
import json
from html.parser import HTMLParser
import string
import random
from multiprocessing import Pool
import time
import itertools
POOL_SIZE = 16


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ' '.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def strip_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remap_keys(mapping):
    return [{'key': k, 'value': v} for k, v in mapping.items()]


def json_reformat(freq_dict, file_name=None):
    if file_name:
        return json.dumps({file_name: remap_keys(dict(freq_dict))})
    return json.dumps(remap_keys(dict(freq_dict)))


nltk.data.path.append('/Users/panda/Desktop/aura_ml/tutorial_model/nltk_data')
post_file_paths = glob('nltk_data/corpora/stack_ex/normalized_files/*')

random.shuffle(post_file_paths)


def chunks(l, n):
    for i in range(0, len(l), int(math.ceil(len(l)/n))):
        yield l[i:i + int(math.ceil(len(l)/n))]


def tokenize(file_path):

    print('New File.')

    def clean(body):
        return strip_punctuation(strip_tags(body)).rstrip().lower()

    tokens = []
    _id = file_path.split('/')[-1]

    text_bodies = [json.loads(row)['body'] for row in open(file_path, 'r')]
    for text_body in text_bodies:
        tokens.extend(clean(text_body).split(' '))
    text_bodies = None
    return tokens


start = time.clock()
with Pool(POOL_SIZE, maxtasksperchild=2) as p:
    tokens_lists = p.map(tokenize, post_file_paths)
tokens = list(itertools.chain.from_iterable(tokens_lists))

freqs = nltk.FreqDist(nltk.bigrams(tokens))
tokens = None

with open('./freqs.json', 'w') as f:
    f.write(json_reformat(freqs))
freqs = None
end = time.clock()

print(end-start)
