import math
import pandas as pd
import numpy as np
from operator import methodcaller
from collections import OrderedDict
from pprint import pprint


class Tokenizer:

    def __init__(self, dataset, embedding_file_path):
        self._dataset = dataset
        self._embedding_file_path = embedding_file_path
        self._build_vocab()
        self._build_reverse_vocab()
        self._build_embedding_matrix()

    def _build_vocab(self):
        def build(word_units):
            word_units = list(set(word_units))
            vocab = OrderedDict()
            for i, v in enumerate(word_units):
                vocab[v] = i
            return vocab

        def get_words(data):
            word_units = []
            for x in data:
                word_units.extend(x.split(' '))
            return word_units

        # all_words = (
        #     get_words(self._dataset['from'])+get_words(self._dataset['to'])
        # )
        all_words = []

        with open(self._embedding_file_path, 'r+') as f:
            next(f)
            for line in f:
                vector_line = line.strip().split(' ')[1:]
                if len(vector_line) != 300:
                    continue
                word = line.split(' ')[0]
                all_words.append(word)
        self._vocab = build(all_words)

    def _build_embedding_matrix(self):
        with open(self._embedding_file_path, 'r+') as f:
            next(f)
            embedding = []
            for line in f:
                vector_line = line.strip().split(' ')[1:]
                if len(vector_line) != 300:
                    continue
                embedding.append(vector_line)
            self._embedding_matrix = np.asarray(embedding, dtype=np.float32)

    def embedding_matrix(self):
        return self._embedding_matrix

    def _build_reverse_vocab(self):
        self._reverse_vocab = {
            _id: key for key, _id in self._vocab.items()
        }

    def reverse_vocab(self):
        return self._reverse_vocab

    def vocab(self):
        return self._vocab

    def batch(self, batch_size, tensor_X_name, tensor_Y_name):
        max_batches = int(math.ceil(len(self._dataset)/batch_size))

        stop = self._vocab['</s>']

        def convert(data, batch_size, index):

            def make_subset(data):
                return list(
                    map(
                        methodcaller('split', ' '),
                        data[(index-1)*batch_size:index*(batch_size)]
                    )
                )
            subset_X = make_subset(data['from'])
            subset_Y = make_subset(data['to'])
            max_X_len = max(map(len, subset_X))
            max_Y_len = max(map(len, subset_Y))

            def process(data, max_len):
                def pad(seq):
                    mapped_seq = [self._vocab.get(y, 54) for y in seq] + [stop]
                    padding = [stop for _ in range(max_len+1-len(mapped_seq))]
                    padded = mapped_seq + padding
                    if len(padded) < 10:
                        padded = padded + [stop for _ in range(10-len(padded))]
                    return padded[:10]

                return list(map(pad, data))

            return process(subset_X, max_X_len), process(subset_Y, max_Y_len)
        while True:
            for x in range(1, max_batches):
                x_seq, y_seq = convert(self._dataset, batch_size, x)
                yield {
                    tensor_X_name: y_seq,
                    tensor_Y_name: x_seq
                }

    def log_formatter(self, keys):
        def to_str_nested(sequence):
            translations = []
            for x in sequence:
                look_up = []
                for val in x:
                    look_up.append(self._reverse_vocab.get(val, "<unk>"))
                translations.append(' '.join(look_up))
            return translations

        def to_str(sequence):
            translation = []
            for x in sequence:
                translation.append(self._reverse_vocab.get(x, "<unk>"))
            return ' '.join(translation)

        def format(values):
            res = []
            for key in keys:
                seq = values[key].tolist()
                if not isinstance(seq[0], list):
                    res.append('{} = {}'.format(key, to_str(seq)))
                else:
                    for x in to_str_nested(seq):
                        res.append('{} = {}'.format(key, to_str_nested(seq)))
            return '\n'.join(res)
        return format
