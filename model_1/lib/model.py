import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.contrib.seq2seq import ScheduledEmbeddingTrainingHelper
from tensorflow.contrib import layers
LSTMCell = tf.nn.rnn_cell.LSTMCell
MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
DropoutWrapper = tf.nn.rnn_cell.DropoutWrapper


class Seq2Seq:
    def __init__(self):
        pass

    def encode(
        self,
        num_units, peepholes, inputs,
        num_layers, seq_len, time_major,
        keep_prob=0.5
    ):
        multi_cell = MultiRNNCell(
            [
               self._cell_factory(num_units, peepholes, keep_prob) for x in range(num_layers)
            ]
        )
        # multi_cell = self._cell_factory(num_units, peepholes, keep_prob)
        # What is the potential benefit of using out_fw over out_bw
        # or the concatentation of the two
        enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multi_cell,
            cell_bw=multi_cell,
            inputs=inputs,
            sequence_length=seq_len,
            dtype=tf.float32,
            time_major=time_major
        )
        return enc_outputs, enc_state

    def _build_param_lists(self, arg_list):
        shared_param_names = [
            'memory_sequence_length',
            'num_units',
            'num_layers',
            'att_depth',
            'att_size',
            'att_multiplier',
            'embedding_matrix',
            'encoder_outputs',
            'vocab_size',
            'batch_size'
        ]
        train_param_names = [
            'encoder_state',
            'sampling_prob',
            'outputs_embedding',
            'outputs_lengths',
        ]
        predict_param_names = [
            'start_tokens',
            'end_token',
            'width',
            'length_penalty',
            'reuse'
        ]

        train_param_names = train_param_names+shared_param_names
        predict_param_names = predict_param_names+shared_param_names

        train_params = {k: arg_list[k] for k in train_param_names}
        predict_params = {k: arg_list[k] for k in predict_param_names}
        return (train_params, predict_params)

    def train(
        self,
        t_out,
        p_out,
        mode,
        outputs,
        optimizer,
        learning_rate,
        summaries,
        batch_size,
        sequence_length
    ):
        tf.identity(t_out.sample_id[0], name='training_predictions')
        weights = tf.Variable(
           initial_value=tf.random_normal([batch_size, 11], dtype=tf.float32),
           dtype=tf.float32,
           trainable=True
        )
        start_tokens = tf.zeros([batch_size], dtype=tf.int64)
        outputs = tf.concat([tf.expand_dims(start_tokens, 1), outputs], 1)

        loss = tf.contrib.seq2seq.sequence_loss(
                t_out.rnn_output, outputs, weights=weights)
        train_op = layers.optimize_loss(
                loss, tf.train.get_global_step(),
                optimizer='Adam',
                learning_rate=learning_rate,
                summaries=['loss', 'learning_rate'])

        tf.identity(p_out.predicted_ids[0], name='predictions')
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=p_out.predicted_ids,
            loss=loss,
            train_op=train_op
        )
    def _cell_factory(self, num_units, peepholes, keep_prob):
        lstm = LSTMCell(num_units=num_units, use_peepholes=peepholes)
        dropout = DropoutWrapper(lstm, input_keep_prob=keep_prob)
        return dropout

    def _build_decoder_cell(
        self, num_units, keep_prob, num_layers, peepholes=True
    ):
        # return self._cell_factory(num_units, peepholes, keep_prob)
        return MultiRNNCell(
            [
               self._cell_factory(num_units, peepholes, keep_prob) for x in range(num_layers)
            ]
        )

    def decode(self, **kwargs):
        decoder_cell = self._build_decoder_cell(
            kwargs['num_units'], kwargs['keep_prob'], kwargs['num_layers']
        )
        train_params, predict_params = self._build_param_lists(kwargs)
        t_out = self._decode_train(cell=decoder_cell, **train_params)
        p_out = self._decode(cell=decoder_cell, **predict_params)
        return t_out[0], p_out[0]

    def _build_attention(
        self, cell,
        encoder_outputs, memory_sequence_length,
        num_units, att_multiplier
    ):
        bahd = BahdanauAttention(
            num_units=num_units,
            memory=encoder_outputs[0],
            memory_sequence_length=memory_sequence_length
        )
        return AttentionWrapper(
            cell=cell,
            attention_mechanism=bahd,
            attention_layer_size=int(num_units*att_multiplier)
        )

    def _decode_train(
        self, outputs_embedding,
        cell, encoder_outputs,
        num_units, num_layers, att_depth,
        att_size, embedding_matrix,
        memory_sequence_length, encoder_state,
        outputs_lengths, batch_size, vocab_size,
        sampling_prob=0.5, att_multiplier=0.5
    ):
        with tf.variable_scope('Decode', reuse=None):
            attention = self._build_attention(
                cell, encoder_outputs, memory_sequence_length, num_units, att_multiplier
            )
            helper = ScheduledEmbeddingTrainingHelper(
                inputs=outputs_embedding,
                embedding=embedding_matrix,
                sequence_length=outputs_lengths,
                sampling_probability=tf.constant(
                    sampling_prob,
                    dtype=tf.float32,
                    name='scheduled_sampling_probability'
                ),
                name='scheduled_sampling_helper'
            )
            output_proj = tf.contrib.rnn.OutputProjectionWrapper(
                attention, vocab_size, reuse=None
            )
            basic_decoder = BasicDecoder(
                cell=output_proj,
                helper=helper,
                initial_state=output_proj.zero_state(
                    batch_size=batch_size,
                    dtype=tf.float32
                ),
            )
            return dynamic_decode(decoder=basic_decoder, output_time_major=False, impute_finished=True)

    def _decode(
        self,
        cell, encoder_outputs, reuse,
        num_units, num_layers, att_depth, vocab_size,
        att_size, embedding_matrix, start_tokens,
        end_token, width, memory_sequence_length,
        batch_size,
        length_penalty=0.0, att_multiplier=0.5
    ):
        with tf.variable_scope('Decode', reuse=True):
            tiled_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, width, 'tiled_batch')
            tiled_lengths = tf.contrib.seq2seq.tile_batch(memory_sequence_length, width, 'tiled_lengths')
            attention = self._build_attention(
                cell, tiled_outputs, tiled_lengths, num_units, att_multiplier
            )
            output_proj = tf.contrib.rnn.OutputProjectionWrapper(
                attention, vocab_size, reuse=True
            )
            beam = BeamSearchDecoder(
                output_proj,
                embedding_matrix,
                tf.to_int32(start_tokens),
                end_token,
                beam_width=width,
                length_penalty_weight=length_penalty,
                initial_state=output_proj.zero_state(
                    batch_size=batch_size*width,
                    dtype=tf.float32
                )
            )
            return dynamic_decode(decoder=beam, output_time_major=False)
