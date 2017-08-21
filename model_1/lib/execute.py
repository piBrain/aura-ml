from sys import argv
from driver import execute
from tokenizer import Tokenizer
import pandas as pd
from pprint import pprint
from evaluate import predict
from functools import partial

DATASET_PATH = './processed_dataset.csv'
EMBEDDING_PATH = './wiki.simple.vec'

tokenizer = Tokenizer(pd.read_csv(DATASET_PATH), EMBEDDING_PATH)

VOCAB = tokenizer.vocab()
VOCAB_SIZE = len(VOCAB)
REVERSE_VOCAB = tokenizer.reverse_vocab()
EMBEDDING = tokenizer.embedding_matrix()
ENCODE_PARAMS = {
    'num_units': 30,
    'num_layers': 1,
    'peepholes': True,
    'keep_probability': 0.7160809299704125,
    'time_major': False
}
DECODE_PARAMS = {
    'num_units': 30,
    'num_layers': 1,
    'attention_depth': 12,
    'attention_size': 3,
    'attention_multiplier': 0.19153857064901542,
    'sampling_probability': 0.4951821610748767,
    'end_token': VOCAB['</s>'],
    'start_tokens': [VOCAB['</st>'] for x in range(20)],
    'beam_width': 30,
    'length_penalty': 0.0,
    'keep_probability': 0.7,
    'reuse': True
}
TRAIN_PARAMS = {
    'optimizer': 'Momentum',
    'learning_rate': 0.03074029999522625,
    'summaries': ['loss', 'learning_rate'],
    'batch_size': 20,
    'data_names': ['input:0', 'output:0'],
    'vocab_size': VOCAB_SIZE,
    'embed_dim': 300,
    'embedding': EMBEDDING,
    'num_steps': 300000,
    'sequence_length': 20
}
PARAMS = {
    'encode': ENCODE_PARAMS,
    'decode': DECODE_PARAMS,
    'train': TRAIN_PARAMS
}

MODEL_DIR = '../saves'

def print_predictions(predictions):
    for pred in predictions:
        print(pred)

def main():
    '''Execute the training or prediction routines of the model.'''

    def feed(feed_func):
        '''Return the next input batch from the feed function.'''
        feature = next(feed_func)
        print(feature)
        return feature
    pprint(argv)
    if argv[1] == 'train':
        feed_function = tokenizer.batch(
            PARAMS['train']['batch_size'],
            PARAMS['train']['data_names'][0],
            PARAMS['train']['data_names'][1]
        )

        execute(PARAMS, MODEL_DIR, partial(feed, feed_function))

    if argv[1] == 'predict':
        feed_function = tokenizer.predict_batch(
            PARAMS['train']['batch_size'],
            PARAMS['train']['data_names'][0],
        )
        predictions = predict(PARAMS, MODEL_DIR, partial(feed, feed_function))
        print_predictions(predictions)
    print('...Done...Exiting!')


if __name__ == "__main__":
    main()
