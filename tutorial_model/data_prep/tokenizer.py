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
        self.unk = self._vocab['</unk>']

    def _build_vocab(self):
        all_words = []

        with open(self._embedding_file_path, 'r+') as f:
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

    def _convert(self, data, batch_size, index):
        start = '<s>'
        stop = '</s>'

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
                mapped_seq = [start] + [y for y in seq] + [stop]
                padding = [stop for _ in range(max_len+1-len(mapped_seq))]
                padded = mapped_seq + padding
                if len(padded) < 20:
                    padded = padded + [stop for _ in range(20-len(padded))]
                return padded[:20]

            return list(map(pad, data))

        return process(subset_X, max_X_len), process(subset_Y, max_Y_len)

    def batch(self, batch_size, tensor_X_name, tensor_Y_name):
        max_batches = int(math.ceil(len(self._dataset)/batch_size))

        while True:
            for x in range(1, max_batches):
                x_seq, y_seq = self._convert(self._dataset, batch_size, x)
                yield {
                    tensor_X_name: x_seq,
                    tensor_Y_name: y_seq
                }

    def predict_batch(self, batch_size, tensor_X_name):
        max_batches = int(math.ceil(len(self._dataset)/batch_size))
        for x in range(1, max_batches):
            x_seq = self._convert(self._dataset, batch_size, x)
            print(x_seq)
            yield {
                tensor_X_name: x_seq,
            }
