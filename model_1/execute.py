import tensorflow as tf
from tensorflow.contrib import layers
from model import Seq2Seq
from tokenizer import Tokenizer
from pprint import pprint
import pandas as pd

DATASET_PATH = './processed_dataset.csv'
EMBEDDING_PATH = './wiki.simple.vec'

tokenizer = Tokenizer(pd.read_csv(DATASET_PATH), EMBEDDING_PATH)

MODEL_SAVE_DIR = './persistance'
VOCAB = tokenizer.vocab()
VOCAB_SIZE = len(VOCAB)
REVERSE_VOCAB = tokenizer.reverse_vocab()
EMBEDDING = tokenizer.embedding_matrix()
ENCODE_PARAMS = {
    'num_units': 300,
    'num_layers': 10,
    'peepholes': True,
    'keep_probability': 0.5,
    'sequence_length': 10,
    'time_major': False
}
DECODE_PARAMS = {
    'num_units': 300,
    'num_layers': 10,
    'attention_depth': 5,
    'attention_size': 5,
    'attention_multiplier': 0.5,
    'sampling_probability': 0.5,
    'end_token': VOCAB['</s>'],
    'beam_width': 5,
    'length_penalty': 0.0,
    'keep_probability': 0.5,
    'reuse': True
}
TRAIN_PARAMS = {
    'optimizer': 'Momentum',
    'learning_rate': 0.3,
    'summaries': ['loss', 'learning_rate'],
    'batch_size': 5,
    'data_names': ['input:0', 'output:0'],
    'vocab_size': VOCAB_SIZE,
    'embed_dim': 300
}
PARAMS = {
    'encode': ENCODE_PARAMS,
    'decode': DECODE_PARAMS,
    'train': TRAIN_PARAMS
}

MODEL_DIR = './saves'


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    print(VOCAB_SIZE)
    estimator = tf.estimator.Estimator(
        model_fn=model_wrapper,
        model_dir=MODEL_DIR,
        params=PARAMS
    )

    def input_fn():
        inputs = tf.placeholder(tf.int32, shape=[None, None], name='input')
        outputs = tf.placeholder(tf.int32, shape=[None, None], name='output')
        tf.identity(inputs[0], 'input_0')
        tf.identity(outputs[0], 'output_0')
        return {
            'input': inputs,
            'output': outputs,
        }, None

    print_inputs = tf.train.LoggingTensorHook(
        PARAMS['train']['data_names'],
        every_n_iter=100,
        formatter=tokenizer.log_formatter(PARAMS['train']['data_names'])
    )

    print_predictions = tf.train.LoggingTensorHook(
        ['predictions', 'training_predictions'], every_n_iter=100,
        formatter=tokenizer.log_formatter(
            ['predictions', 'training_predictions']
        )
    )

    feed_function = tokenizer.batch(
        PARAMS['train']['batch_size'],
        PARAMS['train']['data_names'][0],
        PARAMS['train']['data_names'][1]
    )

    def feed():
        return next(feed_function)

    estimator.train(
        input_fn=input_fn,
        hooks=[tf.train.FeedFnHook(feed), print_inputs, print_predictions],
        steps=1
    )
    print('heyo')


def model_wrapper(features, labels, mode='TRAIN', params=None, config=None):
    model = Seq2Seq()
    inputs = features['input']
    outputs = features['output']
    start_tokens = tf.zeros([params['train']['batch_size']], dtype=tf.int32)
    training_outputs = tf.concat([tf.expand_dims(start_tokens, 1), outputs], 1)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inputs, 1)), 1)
    outputs_lengths = tf.reduce_sum(
        tf.to_int32(tf.not_equal(training_outputs, 1)),
        1
    )
    with tf.variable_scope('embed', reuse=False):
        embeddings = tf.get_variable(
            'embeddings',
            EMBEDDING.shape,
            initializer=tf.constant_initializer(EMBEDDING),
            dtype=tf.float32
        )
    inputs_embedding = layers.embed_sequence(
        inputs,
        vocab_size=params['train']['vocab_size'],
        embed_dim=params['train']['embed_dim'],
        scope='embed',
        reuse=True
    )
    outputs_embedding = layers.embed_sequence(
        training_outputs,
        vocab_size=params['train']['vocab_size'],
        embed_dim=params['train']['embed_dim'],
        scope='embed',
        reuse=True
    )
    encoder_outputs, encoder_state = model.encode(
        num_units=params['encode']['num_units'],
        peepholes=params['encode']['peepholes'],
        inputs=inputs_embedding,
        num_layers=params['encode']['num_layers'],
        seq_len=input_lengths,
        time_major=params['encode']['time_major'],
        keep_prob=params['encode']['keep_probability']
    )
    t_out, t_state, p_out, p_state = model.decode(
        memory_sequence_length=input_lengths,
        num_units=params['decode']['num_units'],
        num_layers=params['decode']['num_layers'],
        att_depth=params['decode']['attention_depth'],
        att_size=params['decode']['attention_size'],
        att_multiplier=params['decode']['attention_multiplier'],
        embedding_matrix=embeddings,
        encoder_outputs=encoder_outputs,
        outputs_embedding=outputs_embedding,
        outputs_lengths=outputs_lengths,
        encoder_state=encoder_state,
        sampling_prob=params['decode']['sampling_probability'],
        start_tokens=start_tokens,
        end_token=params['decode']['end_token'],
        width=params['decode']['beam_width'],
        length_penalty=params['decode']['length_penalty'],
        keep_prob=params['decode']['keep_probability'],
        batch_size=params['train']['batch_size'],
        vocab_size=params['train']['vocab_size'],
        reuse=params['decode']['reuse']
    )
    return model.train(
        t_out,
        p_out,
        outputs,
        optimizer=params['train']['optimizer'],
        learning_rate=params['train']['learning_rate'],
        summaries=params['train']['summaries'],
        batch_size=params['train']['batch_size'],
        sequence_length=input_lengths.shape[0]

    )


if __name__ == "__main__":
    main()
