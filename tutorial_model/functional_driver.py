import tensorflow as tf
# from model.model import Seq2Seq
from tensorflow import estimator
from sys import argv
from pprint import pprint
from tensorflow.python.ops import lookup_ops
# from data_prep.data_hooks import ModelInputs
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.learn import learn_runner, RunConfig, Experiment
from tensorflow.contrib.training import HParams
import json

TRAIN_GRAPH = tf.Graph()
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

class Seq2Seq():

    def __init__(
        self, batch_size, inputs,
        outputs, inp_vocab_size, tgt_vocab_size,
        embed_dim, mode, time_major=False,
        enc_embedding=None, dec_embedding=None, average_across_batch=True,
        average_across_timesteps=True
    ):
        if not enc_embedding:
            self.enc_embedding = tf.contrib.layers.embed_sequence(
                inputs,
                inp_vocab_size,
                embed_dim,
                trainable=True,
                scope='embed'
            )
        else:
            self.enc_embedding = enc_embedding
        if mode == tf.estimator.ModeKeys.TRAIN:
            if not dec_embedding:
                self.dec_embedding = tf.contrib.layers.embed_sequence(
                    outputs,
                    tgt_vocab_size,
                    embed_dim,
                    trainable=True,
                    scope='embed',
                    reuse=True,
                )
            else:
                self.dec_embedding = dec_embedding

        self.inp_vocab_size = inp_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.average_across_batch = average_across_batch
        self.average_across_timesteps = average_across_timesteps
        self.time_major = time_major
        self.batch_size = batch_size

    def _get_lstm(self, num_units):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units)

    def encode(self, num_units, num_layers, seq_len, cell_fw=None, cell_bw=None):
        if cell_fw and cell_bw:
            fw_cell = cell_fw
            bw_cell = cell_bw
        else:
            fw_cell = self._get_lstm(num_units)
            bw_cell = self._get_lstm(num_units)

        encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            self.enc_embedding,
            sequence_length=seq_len,
            time_major=self.time_major,
            dtype=tf.float32
        )
        c_state = tf.concat([bi_encoder_state[0].c, bi_encoder_state[1].c], axis=1)
        h_state = tf.concat([bi_encoder_state[0].h, bi_encoder_state[1].h], axis=1)
        encoder_state = tf.contrib.rnn.LSTMStateTuple(c=c_state, h=h_state)
        return tf.concat(encoder_outputs, -1), encoder_state

    def decode(
        self, num_units, out_seq_len,
        encoder_state, cell=None, helper=None
    ):
        if cell:
            decoder_cell = cell
        else:
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(2*num_units)

        if not helper:
            # dec_helper = ScheduledEmbeddingTrainingHelper(
            #     self.dec_embedding,
            #     embedding=self.embeddings,
            #     seq_len,
            #     sampling_probability,
            # )
            helper = tf.contrib.seq2seq.TrainingHelper(
                self.dec_embedding,
                out_seq_len,
            )
        projection_layer = layers_core.Dense(self.tgt_vocab_size, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            helper,
            encoder_state,
            output_layer=projection_layer
        )
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=20,
            swap_memory=True
        )
        outputs = outputs[0]
        return outputs.rnn_output, outputs.sample_id

    def prepare_predict(self, sample_id):
        return tf.estimator.EstimatorSpec(
            predictions=sample_id
        )

    def prepare_train(
        self, t_out,
        out_seq_len, labels, lr,
        train_op=None, loss=None
    ):
        if not loss:
            weights = tf.sequence_mask(
                out_seq_len,
                20,
                dtype=t_out.dtype
            )
            loss = tf.contrib.seq2seq.sequence_loss(
                t_out,
                labels,
                weights,
                average_across_batch=self.average_across_batch,
                average_across_timesteps=self.average_across_timesteps
            )

        if not train_op:
            train_op = tf.contrib.layers.optimize_loss(
                loss,
                tf.train.get_global_step(),
                optimizer='SGD',
                learning_rate=lr,
                summaries=['loss', 'learning_rate']
            )

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op,
        )

def get_inputs(file_path, vocab_files, batch_size, share_vocab=True, src_eos_id=1, tgt_eos_id=2, num_infer=None, mode=tf.estimator.ModeKeys.TRAIN):
    with tf.name_scope('sentence_markers'):
        src_eos_id = tf.constant(src_eos_id, dtype=tf.int64)
        tgt_eos_id = tf.constant(tgt_eos_id, dtype=tf.int64)
    vocab_tables = _create_vocab_tables(vocab_files, share_vocab)
    if mode == tf.estimator.ModeKeys.TRAIN:
        return _training_input_hook(file_path, vocab_tables, mode, batch_size, src_eos_id, tgt_eos_id)
    if mode == tf.estimator.ModeKeys.EVAL:
        return _validation_input_hook(file_path, vocab_tables, mode, batch_size, src_eos_id, tgt_eos_id)
    if mode == tf.estimator.ModeKeys.PREDICT:
        if num_infer is None:
            raise ArgumentError('If performing inference must supply number of predictions to be made.')
        return _infer_input_hook(file_path, num_infer, vocab_tables, batch_size, src_eos_id, tgt_eos_id)

def _prepare_data(dataset, vocab_tables, out=False):
    prep_set = dataset.map(lambda string: tf.string_split([string]).values)
    prep_set = prep_set.map(lambda words: (words, tf.size(words)))
    if out == True:
        return prep_set.map(lambda words, size: (vocab_tables[1].lookup(words), size))
    return prep_set.map(lambda words, size: (vocab_tables[0].lookup(words), size))

def _batch_data(dataset, batch_size, mode, src_eos_id, tgt_eos_id):
    batched_set = dataset.padded_batch(
            batch_size,
            padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([]))),
            padding_values=((src_eos_id, 0), (tgt_eos_id, 0))
    )
    return batched_set

def _create_vocab_tables(vocab_files, share_vocab=False):
    if vocab_files[1] is None and share_vocab == False:
        raise ArgumentError('If share_vocab is set to false must provide target vocab. (src_vocab_file, \
                target_vocab_file)')

    src_vocab_table = lookup_ops.index_table_from_file(
        vocab_files[0],
        default_value=UNK_ID
    )

    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
            vocab_files[1],
            default_value=UNK_ID
        )

    return src_vocab_table, tgt_vocab_table

def _create_iterator_hook(scope_name, mode, iterator, file_path, name_placeholder):
    hook = IteratorInitializerHook()
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        feed_dict = {
                name_placeholder[0]: file_path[0],
                name_placeholder[1]: file_path[1]
        }
    else:
        feed_dict = {name_placeholder: file_path}

    with tf.name_scope(scope_name):
        hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict=feed_dict,
                )
    return hook

def _set_up_train_or_eval(scope_name, mode, vocab_tables, batch_size, file_path, src_eos_id, tgt_eos_id):
    with tf.name_scope(scope_name):
        in_file = tf.placeholder(tf.string, shape=())
        in_dataset = _prepare_data(tf.contrib.data.TextLineDataset(in_file), vocab_tables)
        out_file = tf.placeholder(tf.string, shape=())
        out_dataset = _prepare_data(tf.contrib.data.TextLineDataset(out_file), vocab_tables)
        dataset = tf.contrib.data.Dataset.zip((in_dataset, out_dataset))
        dataset = _batch_data(dataset, mode, batch_size, src_eos_id, tgt_eos_id)
        iterator = dataset.make_initializable_iterator()

    hook = _create_iterator_hook(scope_name, mode, iterator, file_path, (in_file, out_file))

    return (iterator, hook)

def _training_input_hook(file_path, vocab_tables, batch_size, mode, src_eos_id, tgt_eos_id):
    iterator, hook = _set_up_train_or_eval('train_inputs', mode, vocab_tables, batch_size, file_path, src_eos_id, tgt_eos_id)

    return (iterator, hook)

def _validation_input_hook(file_path, vocab_tables, batch_size, mode, src_eos_id, tgt_eos_id):
    iterator, hook = _set_up_train_or_eval('eval_inputs', mode, vocab_tables, batch_size, file_path, src_eos_id, tgt_eos_id)

    return (iterator, hook)

def _infer_input_hook(self, file_path, num_infer, mode, vocab_tables, batch_size, src_eos_id, tgt_eos_id):
    with tf.name_scope('infer_inputs'):
        infer_file = tf.placeholder(tf.string, shape=(num_infer))
        dataset = tf.contrib.data.TextLineDataset(infer_file)
        dataset = _prepare_data(dataset)
        dataset = _batch_data(dataset, mode, batch_size, src_eos_id, tgt_eos_id)
        iterator = dataset.make_initalizable_iterator()

    hook = _create_iterator_hook('infer_inputs', mode, iterator, file_path, infer_file)

    return (iterator, hook)


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

with open('./hyperparameters.json', 'r') as f:
    HPARAMS = json.load(f)

def model_fn(features, labels, mode, params, config):
    model = Seq2Seq(
            params.batch_size, features[0], labels[0],
            params.input_vocab_size, params.output_vocab_size, params.num_units,
            mode
    )

    enc_out, enc_state = model.encode(params.num_units, params.num_layers, features[1])

    if mode == estimator.ModeKeys.TRAIN:
        t_out, _ = model.decode(params.num_units, labels[1], enc_state)
        spec = model.prepare_train(t_out, labels[1], labels[0], params.learning_rate)
    if mode == estimator.ModeKeys.PREDICT:
        _, sample_id = model.decode(params.num_units, labels[1])
        spec = model.prepare_predict(sample_id)
    return spec

def experiment_fn(run_config, hparams):
    train_iter, train_input_hook = get_inputs(hparams.train_dataset_paths, hparams.vocab_paths, hparams.batch_size)
    eval_iter, eval_input_hook = get_inputs(hparams.eval_dataset_paths, hparams.vocab_paths, hparams.batch_size, mode=estimator.ModeKeys.EVAL)
    def train_input_fn():
        return train_iter.get_next()
    def eval_input_fn():
        return eval_iter.get_next()

    exp_estimator = get_estimator(run_config, hparams)
    run_config.replace(save_checkpoints_steps=hparams.min_eval_frequency)

    return Experiment(
        estimator=exp_estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=hparams.num_steps,
        min_eval_frequency=hparams.min_eval_frequency,
        train_monitors=[train_input_hook],
        eval_hooks=[eval_input_hook],
        eval_steps=None
    )

def get_estimator(run_config, hparams):
    return estimator.Estimator(
        model_fn=model_fn,
        params=hparams,
        config=run_config,
    )

def print_predictions(predictions):
    for pred in predictions:
        pprint(pred)

def main():
    hparams = HParams(**HPARAMS)
    run_config = RunConfig(model_dir='./save')

    if len(argv) < 2 or argv[1] == 'train':
        learn_runner.run(
            experiment_fn=experiment_fn,
            run_config=run_config,
            schedule="train_and_evaluate",
            hparams=hparams,
        )
    elif argv[1] == 'predict':
        pass
    else:
        print('Unknown Operation.')

if __name__ == '__main__':
    main()
