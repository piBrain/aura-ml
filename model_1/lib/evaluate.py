import tensorflow as tf
from tensorflow.contrib import layers
from model import Seq2Seq


def model_wrapper(features, labels, mode, params=None, config=None):
    return model.train(
        t_out,
        p_out,
        mode,
        outputs,
        optimizer=params['train']['optimizer'],
        learning_rate=params['train']['learning_rate'],
        summaries=params['train']['summaries'],
        batch_size=params['train']['batch_size'],
        sequence_length=input_lengths.shape[0]

    )
def predict(params, model_dir, feed_fn, patch_loss_return=False):
    tf.logging.set_verbosity(tf.logging.INFO)
    def input_fn():
        inputs = tf.placeholder(tf.int64, shape=[params['train']['batch_size'], params['train']['sequence_length']], name='input')
        tf.identity(inputs[0], 'input_0')
        return inputs
    estimator = tf.estimator.Estimator(
        model_fn=model_wrapper,
        model_dir=model_dir,
        params=params
    )
    return estimator.predict(
        input_fn=input_fn,
        hooks=[tf.train.FeedFnHook(feed_fn)],
    )
