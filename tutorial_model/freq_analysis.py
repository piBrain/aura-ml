import nltk
import math
from glob import glob
from pprint import pprint
import json
from html.parser import HTMLParser
import string
import random
from multiprocessing import Pool

POOL_SIZE = 4

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


def analyse(file_paths):
    print("Starting!")

    def clean(body):
        return strip_punctuation(strip_tags(body)).rstrip().lower()

    freqs = nltk.FreqDist()
    freqs_by_text = {}

    for path in file_paths:
        print('New File.')
        _id = path.split('/')[-1]
        freqs_by_text[_id] = nltk.FreqDist()
        text_bodies = [json.loads(row)['body'] for row in open(path, 'r')]
        print(len(text_bodies))
        for text_body in text_bodies:
            tokens = clean(text_body).split(' ')
            freqs += nltk.FreqDist(nltk.bigrams(tokens))
            freqs_by_text[_id] += nltk.FreqDist(nltk.bigrams(tokens))
    return (freqs, freqs_by_text)


chunked_paths = [chunk for chunk in chunks(post_file_paths, POOL_SIZE)]

with Pool(POOL_SIZE) as p:
    multi_returns = p.map(analyse, chunked_paths)
    final_freqs = nltk.FreqDist()
    final_freqs_by_text = {}
    for freq_tups in multi_returns:
        final_freqs += freq_tups[0]
        final_freqs_by_text = dict(final_freqs_by_text, **freq_tups[1])

    with open('./freqs_by_text.json', 'w') as f:
        for k, v in final_freqs_by_text.items():
            f.write(json_reformat(v, k))

    with open('./freqs.json', 'w') as f:
        f.write(json_reformat(final_freqs))
