from driver import execute
from tokenizer import Tokenizer
from pprint import pprint
from tempfile import TemporaryDirectory
import pandas as pd
import hyperopt
import time
import json
from bson import json_util 

DATASET_PATH = './processed_dataset.csv'
EMBEDDING_PATH = './wiki.simple.vec'


tokenizer = Tokenizer(pd.read_csv(DATASET_PATH), EMBEDDING_PATH)

VOCAB = tokenizer.vocab()
VOCAB_SIZE = len(VOCAB)
REVERSE_VOCAB = tokenizer.reverse_vocab()
EMBEDDING = tokenizer.embedding_matrix()

def objective(args):

    ENCODE_PARAMS = {
        'num_units': args['num_units'],
        'num_layers': args['num_layers'],
        'peepholes': True,
        'keep_probability': args['keep_probability'],
        'time_major': False
    }
    DECODE_PARAMS = {
        'num_units': args['num_units'],
        'num_layers': args['num_layers'],
        'attention_depth': args['attention_depth'],
        'attention_size': args['attention_multiplier'],
        'attention_multiplier': args['attention_multiplier'],
        'sampling_probability': args['sampling_probability'],
        'end_token': VOCAB['</s>'],
        'start_tokens': [VOCAB['</st>'] for x in range(args['batch_size'])],
        'beam_width': args['beam_width'],
        'length_penalty': 0.0,
        'keep_probability': args['keep_probability'],
        'reuse': True
    }
    TRAIN_PARAMS = {
        'optimizer': 'Momentum',
        'learning_rate': args['learning_rate'],
        'summaries': ['loss', 'learning_rate'],
        'batch_size': args['batch_size'],
        'data_names': ['input:0', 'output:0'],
        'vocab_size': VOCAB_SIZE,
        'embed_dim': 300,
        'embedding': EMBEDDING,
        'num_steps': 10000,
        'sequence_length': 20
    }
    PARAMS = {
        'encode': ENCODE_PARAMS,
        'decode': DECODE_PARAMS,
        'train': TRAIN_PARAMS
    }

    feed_function = tokenizer.batch(
        PARAMS['train']['batch_size'],
        PARAMS['train']['data_names'][0],
        PARAMS['train']['data_names'][1]
    )

    def feed():
        return next(feed_function)

    with TemporaryDirectory() as temp_dir:
        try:
            return { 'loss': execute(PARAMS, temp_dir, feed, True), 'status': hyperopt.STATUS_OK, 'eval_time': time.time(), 'params': args }
        except:
            return { 'loss': 100000000001, 'status': hyperopt.STATUS_OK, 'eval_time': time.time(), 'params': args }
    

def optimize():

    space = {
        'num_layers': hyperopt.hp.choice('num_layers', [1, 5, 7, 10, 12, 15, 20]),
        'num_units': hyperopt.hp.choice('num_units', [30, 100, 150, 200, 300]),
        'keep_probability': hyperopt.hp.choice('keep_probability', [hyperopt.hp.uniform('kpprb', 0.0, 1.0)]),
        'attention_depth': hyperopt.hp.choice('attention_depth', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20]),
        'attention_size': hyperopt.hp.choice('attention_size', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20]),
        'attention_multiplier': hyperopt.hp.choice('attention_multiplier', [hyperopt.hp.uniform('attml', 0.0, 1.0)]),
        'sampling_probability': hyperopt.hp.choice('sampling_probability', [hyperopt.hp.uniform('smp', 0.0, 1.0)]),
        'beam_width': hyperopt.hp.choice('beam_width', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 30, 40, 50]),
        'learning_rate': hyperopt.hp.choice('learning_rate', [hyperopt.hp.uniform('lr', 0.0, 0.5)]),
        'batch_size': hyperopt.hp.choice('batch_size', [20])
    }
    trials = hyperopt.Trials()
    best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=100, trials=trials)

    def get_loss(param_dict):
        return param_dict['result']['loss']

    print('------------------BEST-----------------')
    print(best_model)
    print('---------------------------------------')
    print('------------------SPACE----------------')
    print(hyperopt.space_eval(space, best_model))
    print('--------------------------------------')
    sorted_trials = sorted(trials.trials, key=get_loss)
    with open('./opt_param_list.json', 'w') as f:
        for trial in sorted_trials:
            json.dump(trial, f, default=json_util.default)
    
if __name__ == '__main__':
    optimize()
