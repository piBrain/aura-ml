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
from gensim.models import KeyedVectors
import numpy as np
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
        average_across_timesteps=True, vocab_path=None, embedding_path='./data_files/wiki.simple.vec'
    ):
        embed_np = self._get_embedding(embedding_path)
        if not enc_embedding:
            self.enc_embedding = tf.contrib.layers.embed_sequence(
                inputs,
                inp_vocab_size,
                embed_dim,
                trainable=True,
                scope='embed',
                initializer=tf.constant_initializer(value=embed_np, dtype=tf.float32)
            )
        else:
            self.enc_embedding = enc_embedding
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            if not dec_embedding:
                embed_outputs = tf.contrib.layers.embed_sequence(
                    outputs,
                    tgt_vocab_size,
                    embed_dim,
                    trainable=True,
                    scope='embed',
                    reuse=True
                )
                with tf.variable_scope('embed', reuse=True):
                    dec_embedding = tf.get_variable('embeddings')
                self.embed_outputs = embed_outputs
                self.dec_embedding = dec_embedding

            else:
                self.dec_embedding = dec_embedding
        else:
            with tf.variable_scope('embed', reuse=True):
                self.dec_embedding = tf.get_variable('embeddings')

        if mode == tf.estimator.ModeKeys.PREDICT and vocab_path is None:
            raise ValueError('If mode is predict, must supply vocab_path')
        self.vocab_path = vocab_path
        self.inp_vocab_size = inp_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.average_across_batch = average_across_batch
        self.average_across_timesteps = average_across_timesteps
        self.time_major = time_major
        self.batch_size = batch_size
        self.mode = mode

    def _get_embedding(self, embedding_path):
        model = KeyedVectors.load_word2vec_format(embedding_path)
        vocab = model.vocab
        vocab_len = len(vocab)
        return np.array([model.word_vec(k) for k in vocab.keys()])

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

    def _train_decoder(self, decoder_cell, out_seq_len, encoder_state, helper):
        if not helper:
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                self.embed_outputs,
                out_seq_len,
                self.dec_embedding,
                0.6,
            )
            # helper = tf.contrib.seq2seq.TrainingHelper(
            #     self.dec_embedding,
            #     out_seq_len,
            # )
        projection_layer = layers_core.Dense(self.tgt_vocab_size, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            helper,
            encoder_state,
            output_layer=projection_layer
        )
        return decoder

    def _predict_decoder(self, cell, encoder_state, beam_width, length_penalty_weight):
        with tf.name_scope('sentence_markers'):
            sos_id = tf.constant(1, dtype=tf.int32)
            eos_id = tf.constant(2, dtype=tf.int32)
        start_tokens = tf.fill([self.batch_size], sos_id)
        end_token = eos_id
        projection_layer = layers_core.Dense(self.tgt_vocab_size, use_bias=False)
        emb = tf.squeeze(self.dec_embedding)
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cell,
            embedding=self.dec_embedding,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=encoder_state,
            beam_width=beam_width,
            output_layer=projection_layer,
            length_penalty_weight=length_penalty_weight
        )
        return decoder

    def _attention(self, num_units, memory, memory_sequence_length, beam_width=None):
        if beam_width:
            memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
            memory_sequence_length = tf.contrib.seq2seq.tile_batch(memory_sequence_length, multiplier=beam_width)
        return tf.contrib.seq2seq.BahdanauAttention(num_units, memory, memory_sequence_length)

    def decode(
        self, num_units, out_seq_len, in_seq_len, encoder_outputs,
        encoder_state, cell=None, helper=None,
        beam_width=None, length_penalty_weight=None
    ):
        with tf.name_scope('Decode'):
            if cell:
                decoder_cell = cell
            else:
                decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(2*num_units)
                if beam_width:
                    print(beam_width)
                    attention_mechanism = self._attention(2*num_units, encoder_outputs, in_seq_len, beam_width)
                else:
                    attention_mechanism = self._attention(2*num_units, encoder_outputs, in_seq_len)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell,
                    attention_mechanism,
                    attention_layer_size=2*num_units,
                    alignment_history=False,
                    name='attention_wrapper'
                )
            if self.mode != estimator.ModeKeys.PREDICT:
                attn_encoder_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state)
                decoder = self._train_decoder(decoder_cell, out_seq_len, attn_encoder_state, helper)
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    maximum_iterations=20,
                    impute_finished=True,
                    swap_memory=True,
                )
            else:
                tiled_encoder_state = tf.contrib.seq2seq.tile_batch(
                    encoder_state, multiplier=beam_width
                )
                # tiled_encoder_state = tf.nn.rnn_cell.LSTMStateTuple(
                #     tf.contrib.seq2seq.tile_batch(encoder_state[0], multipler=beam_width),
                #     tf.contrib.seq2seq.tile_batch(encoder_state[1], multipler=beam_width)
                # )
                attn_encoder_state = decoder_cell.zero_state(self.batch_size*beam_width, tf.float32).clone(cell_state=tiled_encoder_state)
                decoder = self._predict_decoder(decoder_cell, attn_encoder_state, beam_width, length_penalty_weight)
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    maximum_iterations=20,
                    swap_memory=True,
                )
            outputs = outputs[0]
            if self.mode != estimator.ModeKeys.PREDICT:
                return outputs.rnn_output, outputs.sample_id
            else:
                return outputs.beam_search_decoder_output, outputs.predicted_ids

    def prepare_predict(self, sample_id):
        rev_table = lookup_ops.index_to_string_table_from_file(
            self.vocab_path, default_value=UNK)
        predictions = rev_table.lookup(tf.to_int64(sample_id))
        return tf.estimator.EstimatorSpec(
            predictions=predictions,
            mode=tf.estimator.ModeKeys.PREDICT
        )

    def prepare_train_eval(
        self, t_out,
        out_seq_len, labels, lr,
        train_op=None, loss=None
    ):
        if not loss:
            weights = tf.sequence_mask(
                out_seq_len,
                dtype=t_out.dtype
            )
            loss = tf.contrib.seq2seq.sequence_loss(
                t_out,
                labels,
                weights,
                average_across_batch=self.average_across_batch,
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
            mode=self.mode,
            loss=loss,
            train_op=train_op,
        )

class ModelInputs(object):
    """Factory to construct various input hooks and functions depending on mode """

    def __init__(
        self, vocab_files, batch_size,
        share_vocab=True, src_eos_id=1, tgt_eos_id=2
    ):
        self.batch_size = batch_size
        self.vocab_files = vocab_files
        self.share_vocab = share_vocab
        self.src_eos_id = src_eos_id
        self.tgt_eos_id = tgt_eos_id

    def get_inputs(self, file_path, num_infer=None, mode=tf.estimator.ModeKeys.TRAIN):
        self.mode = mode
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            return self._training_input_hook(file_path)
        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self._validation_input_hook(file_path)
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            if num_infer is None:
                raise ValueError('If performing inference must supply number of predictions to be made.')
            return self._infer_input_hook(file_path, num_infer)

    def _prepare_data(self, dataset, out=False):
        prep_set = dataset.map(lambda string: tf.string_split([string]).values)
        prep_set = prep_set.map(lambda words: (words, tf.size(words)))
        if out == True:
            return prep_set.map(lambda words, size: (self.vocab_tables[1].lookup(words), size))
        return prep_set.map(lambda words, size: (self.vocab_tables[0].lookup(words), size))

    def _batch_data(self, dataset, src_eos_id, tgt_eos_id):
        batched_set = dataset.padded_batch(
                self.batch_size,
                padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([]))),
                padding_values=((src_eos_id, 0), (tgt_eos_id, 0))
        )
        return batched_set

    def _batch_infer_data(self, dataset, src_eos_id):
        batched_set = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
            padding_values=(src_eos_id, 0)
        )
        return batched_set

    def _create_vocab_tables(self, vocab_files, share_vocab=False):
        if vocab_files[1] is None and share_vocab == False:
            raise ValueError('If share_vocab is set to false must provide target vocab. (src_vocab_file, \
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

    def _prepare_iterator_hook(self, hook, scope_name, iterator, file_path, name_placeholder):
        if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
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

    def _set_up_train_or_eval(self, scope_name, file_path):
        hook = IteratorInitializerHook()
        def input_fn():
            with tf.name_scope(scope_name):
                with tf.name_scope('sentence_markers'):
                    src_eos_id = tf.constant(self.src_eos_id, dtype=tf.int64)
                    tgt_eos_id = tf.constant(self.tgt_eos_id, dtype=tf.int64)
                self.vocab_tables = self._create_vocab_tables(self.vocab_files, self.share_vocab)
                in_file = tf.placeholder(tf.string, shape=())
                in_dataset = self._prepare_data(tf.contrib.data.TextLineDataset(in_file).repeat(None))
                out_file = tf.placeholder(tf.string, shape=())
                out_dataset = self._prepare_data(tf.contrib.data.TextLineDataset(out_file).repeat(None))
                dataset = tf.contrib.data.Dataset.zip((in_dataset, out_dataset))
                dataset = self._batch_data(dataset, src_eos_id, tgt_eos_id)
                iterator = dataset.make_initializable_iterator()
                next_example, next_label = iterator.get_next()
                self._prepare_iterator_hook(hook, scope_name, iterator, file_path, (in_file, out_file))
                return next_example, next_label

        return (input_fn, hook)

    def _training_input_hook(self, file_path):
        input_fn, hook = self._set_up_train_or_eval('train_inputs', file_path)

        return (input_fn, hook)

    def _validation_input_hook(self, file_path):
        input_fn, hook = self._set_up_train_or_eval('eval_inputs', file_path)

        return (input_fn, hook)

    def _infer_input_hook(self, file_path, num_infer):
        hook = IteratorInitializerHook()

        def input_fn():
            with tf.name_scope('infer_inputs'):
                with tf.name_scope('sentence_markers'):
                    src_eos_id = tf.constant(self.src_eos_id, dtype=tf.int64)
                self.vocab_tables = self._create_vocab_tables(self.vocab_files, self.share_vocab)
                infer_file = tf.placeholder(tf.string, shape=())
                dataset = tf.contrib.data.TextLineDataset(infer_file)
                dataset = self._prepare_data(dataset)
                dataset = self._batch_infer_data(dataset, src_eos_id)
                iterator = dataset.make_initializable_iterator()
                next_example, seq_len = iterator.get_next()
                self._prepare_iterator_hook(hook, 'infer_inputs', iterator, file_path, infer_file)
                return ((next_example, seq_len), None)

        return (input_fn, hook)


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
    if labels:
        label_data = labels[0]
        out_seq_len = labels[1]
    else:
        label_data = None
        out_seq_len = None
    model = Seq2Seq(
            params.batch_size, features[0], label_data,
            params.input_vocab_size, params.output_vocab_size, params.num_units*10,
            mode, vocab_path=params.vocab_paths[0]
    )

    enc_out, enc_state = model.encode(params.num_units, params.num_layers, features[1])

    if mode == estimator.ModeKeys.TRAIN or mode == estimator.ModeKeys.EVAL:
        t_out, _ = model.decode(params.num_units, out_seq_len, features[1], enc_out, enc_state)
        spec = model.prepare_train_eval(t_out, out_seq_len, label_data, params.learning_rate)
    if mode == estimator.ModeKeys.PREDICT:
        _, sample_id = model.decode(params.num_units, out_seq_len, features[1], enc_out, enc_state, beam_width=params.beam_width,
                length_penalty_weight=params.length_penalty_weight)
        spec = model.prepare_predict(sample_id)
    return spec

def _set_up_infer():
    pass

def experiment_fn(run_config, hparams):
    input_fn_factory = ModelInputs(hparams.vocab_paths, hparams.batch_size)
    train_input_fn, train_input_hook = input_fn_factory.get_inputs(hparams.train_dataset_paths)
    eval_input_fn, eval_input_hook = input_fn_factory.get_inputs(hparams.eval_dataset_paths, mode=estimator.ModeKeys.EVAL)

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
        eval_steps=1000
    )

def get_estimator(run_config, hparams):
    return estimator.Estimator(
        model_fn=model_fn,
        params=hparams,
        config=run_config,
    )

def print_predictions(predictions, hparams):
    for pred in predictions:
        for sent in pred:
            stred = map(str, sent)
            print(' '.join(stred))

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
        input_fn_factory = ModelInputs(hparams.vocab_paths, hparams.batch_size)
        predict_input_fn, predict_input_hook = input_fn_factory.get_inputs(hparams.predict_dataset_path,
                mode=estimator.ModeKeys.PREDICT, num_infer=20)
        classifier = get_estimator(run_config, hparams)
        predictions = classifier.predict(input_fn=predict_input_fn, hooks=[predict_input_hook])
        print_predictions(predictions, hparams)
    else:
        print('Unknown Operation.')

if __name__ == '__main__':
    main()
