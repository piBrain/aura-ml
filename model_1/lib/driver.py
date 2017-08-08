import tensorflow as tf
from tensorflow.contrib import layers
from model import Seq2Seq

def model_wrapper(features, labels, mode, params=None, config=None):
    model = Seq2Seq()
    inputs = features
    outputs = labels
    batch_size = tf.shape(inputs)[0]
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inputs, params['train']['vocab_size'])), 1, name='input_lengths')
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(outputs, params['train']['vocab_size'])), 1, name='output_lengths')
    inputs_embedding = layers.embed_sequence(
        inputs,
        vocab_size=params['train']['vocab_size'],
        embed_dim=params['train']['embed_dim'],
        initializer=tf.constant_initializer(params['train']['embedding']),
        trainable=True,
        scope='embed',
    )
    outputs_embedding = layers.embed_sequence(
        outputs,
        vocab_size=params['train']['vocab_size'],
        embed_dim=params['train']['embed_dim'],
        trainable=True,
        scope='embed',
        reuse=True,
    )
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')
    encoder_outputs, encoder_state = model.encode(
        num_units=params['encode']['num_units'],
        peepholes=params['encode']['peepholes'],
        inputs=inputs_embedding,
        num_layers=params['encode']['num_layers'],
        seq_len=input_lengths,
        time_major=params['encode']['time_major'],
        keep_prob=params['encode']['keep_probability']
    )
    t_out, p_out = model.decode(
        memory_sequence_length=input_lengths,
        num_units=params['decode']['num_units'],
        num_layers=params['decode']['num_layers'],
        att_depth=params['decode']['attention_depth'],
        att_size=params['decode']['attention_size'],
        att_multiplier=params['decode']['attention_multiplier'],
        embedding_matrix=embeddings,
        encoder_outputs=encoder_outputs,
        outputs_embedding=outputs_embedding,
        outputs_lengths=output_lengths,
        encoder_state=encoder_state,
        sampling_prob=params['decode']['sampling_probability'],
        start_tokens=params['decode']['start_tokens'],
        end_token=params['decode']['end_token'],
        width=params['decode']['beam_width'],
        length_penalty=params['decode']['length_penalty'],
        keep_prob=params['decode']['keep_probability'],
        batch_size=batch_size,
        vocab_size=params['train']['vocab_size'],
        reuse=params['decode']['reuse']
    )
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

def monkeypatch_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

def execute(params, model_dir, feed_fn, patch_loss_return=False):
    tf.logging.set_verbosity(tf.logging.INFO)
    def input_fn():
        inputs = tf.placeholder(tf.int64, shape=[params['train']['batch_size'], params['train']['sequence_length']], name='input')
        outputs = tf.placeholder(tf.int64, shape=[params['train']['batch_size'], params['train']['sequence_length']], name='output')
        tf.identity(inputs[0], 'input_0')
        tf.identity(outputs[0], 'output_0')
        return inputs, outputs
    if patch_loss_return:
        from tensorflow.python.training import saver
        from tensorflow.python.training import training
        from tensorflow.python.platform import tf_logging as logging
        @monkeypatch_method(tf.estimator.Estimator)
        def train(self, input_fn, hooks=None, steps=None, max_steps=None):
            def _check_hooks_type(hooks):
                hooks = list(hooks or [])
                for h in hooks:
                    if not isinstance(h, training.SessionRunHook):
                        raise TypeError('Hooks must be a SessionRunHook, given: {}'.format(h))
                return hooks

            if (steps is not None) and (max_steps is not None):
                raise ValueError('Can not provide both steps and max_steps.')
            if steps is not None and steps <= 0:
                raise ValueError('Must specify steps > 0, given: {}'.format(steps))
            if max_steps is not None and max_steps <= 0:
                raise ValueError('Must specify max_steps > 0, given: {}'.format(max_steps))

            if max_steps is not None:
                start_step = _load_global_step_from_checkpoint_dir(self._model_dir)
                if max_steps <= start_step:
                    logging.info('Skipping training since max_steps has already saved.')
                    return self

            hooks = _check_hooks_type(hooks)
            if steps is not None or max_steps is not None:
                hooks.append(training.StopAtStepHook(steps, max_steps))

            loss = self._train_model(input_fn=input_fn, hooks=hooks)
            tf.logging.info('Loss for final step: %s.', loss)
            return self, loss

    estimator = tf.estimator.Estimator(
        model_fn=model_wrapper,
        model_dir=model_dir,
        params=params
    )

    # print_predictions = tf.train.LoggingTensorHook(
    #     ['predictions', 'training_predictions'], every_n_iter=10000,
    #     formatter=tokenizer.log_formatter(
    #         ['predictions', 'training_predictions']
    #     )
    # )
    if patch_loss_return:
        return estimator.train(
            input_fn=input_fn,
            hooks=[tf.train.FeedFnHook(feed_fn)],
            steps=params['train']['num_steps']
        )[1]

    return estimator.train(
        input_fn=input_fn,
        hooks=[tf.train.FeedFnHook(feed_fn)],
        steps=params['train']['num_steps']
    )

