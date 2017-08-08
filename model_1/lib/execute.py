from driver import execute
from tokenizer import Tokenizer
import pandas as pd
from pprint import pprint

DATASET_PATH = './processed_dataset.csv'
EMBEDDING_PATH = './wiki.simple.vec'

tokenizer = Tokenizer(pd.read_csv(DATASET_PATH), EMBEDDING_PATH)

VOCAB = tokenizer.vocab()
VOCAB_SIZE = len(VOCAB)
REVERSE_VOCAB = tokenizer.reverse_vocab()
EMBEDDING = tokenizer.embedding_matrix()
ENCODE_PARAMS = {
    'num_units': 30,
    'num_layers': 10,
    'peepholes': True,
    'keep_probability': 0.7,
    'time_major': False
}
DECODE_PARAMS = {
    'num_units': 30,
    'num_layers': 9,
    'attention_depth': 2,
    'attention_size': 3,
    'attention_multiplier': 0.7,
    'sampling_probability': 0.6,
    'end_token': VOCAB['</s>'],
    'start_tokens': [VOCAB['</st>'] for x in range(20)],
    'beam_width': 6,
    'length_penalty': 0.0,
    'keep_probability': 0.7,
    'reuse': True
}
TRAIN_PARAMS = {
    'optimizer': 'Momentum',
    'learning_rate': 0.01,
    'summaries': ['loss', 'learning_rate'],
    'batch_size': 20,
    'data_names': ['input:0', 'output:0'],
    'vocab_size': VOCAB_SIZE,
    'embed_dim': 300,
    'embedding': EMBEDDING,
    'num_steps': 100,
    'sequence_length': 20
}
PARAMS = {
    'encode': ENCODE_PARAMS,
    'decode': DECODE_PARAMS,
    'train': TRAIN_PARAMS
}

MODEL_DIR = '../saves'

def main():
    feed_function = tokenizer.batch(
        PARAMS['train']['batch_size'],
        PARAMS['train']['data_names'][0],
        PARAMS['train']['data_names'][1]
    )

    def feed():
        return next(feed_function)

    execute(PARAMS, MODEL_DIR, feed)

if __name__ == "__main__":
    main()
