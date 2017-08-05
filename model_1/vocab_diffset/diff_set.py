import math
import pandas as pd
import numpy as np
from operator import methodcaller
from collections import OrderedDict
from pprint import pprint


def fb_vocab(vocab=set()):
    if vocab:
        return vocab
    all_words = []
    with open('wiki.simple.vec', 'r+') as f:
        next(f)
        for line in f:
            vector_line = line.strip().split(' ')[1:]
            if len(vector_line) != 300:
                continue
            word = line.split(' ')[0]
            all_words.append(word)
    vocab.update(all_words)
    return vocab


def dataset_vocab(vocab=set()):
    if vocab:
        return vocab
    def get_words(data):
        word_units = []
        for x in data:
            word_units.extend(x.split(' '))
        return word_units
    all_words = []
    dataset = pd.read_csv('./processed_dataset.csv')
    all_words.extend(get_words(dataset['to']))
    all_words.extend(get_words(dataset['from']))
    vocab.update(all_words)
    return vocab


def main():
    for x in dataset_vocab().difference(fb_vocab()):
        print(x)

if __name__ == "__main__":
    main()


