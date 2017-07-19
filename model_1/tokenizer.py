import math
import pandas as pd
import numpy as np
from functools import partial
from collections import OrderedDict


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

        def convert(data, batch_size, index):
            stop = self._vocab['</s>']

            def make_subset(data):
                return list(
                    map(
                        partial(str.split, ' '),
                        data[(index-1)*batch_size:index*(batch_size)]
                    )
                )
            subset_X = make_subset(data['from'])
            subset_Y = make_subset(data['to'])
            max_len = max(max(map(len, subset_X)), max(map(len, subset_Y)))

            def process(data):
                def pad(seq):
                    mapped_seq = [self._vocab.get(y, 54) for y in seq]
                    return mapped_seq + [stop*(max_len-len(mapped_seq))]

                return list(map(pad, data))

            return process(subset_X), process(subset_Y)

        for x in range(1, max_batches):
            x_seq, y_seq = convert(self._dataset, batch_size, x)
            yield {
                tensor_X_name: x_seq,
                tensor_Y_name: y_seq
            }

    def log_formatter(self, keys):
        def to_str(sequence):
            tokens = [
                self._reverse_vocab.get(x, "<unk>") for x in sequence
            ]
            return ' '.join(tokens)

        def format(values):
            res = []
            for key in keys:
                res.append('{} = {}'.format(key, to_str(values[key])))
            return '\n'.join(res)
        return format
