import tensorflow as tf
from tensorflow.python.layers import core as layers_core

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
                    reuse=True
                )
            else:
                self.dec_embedding = dec_embedding

        with tf.variable_scope('embed', reuse=True):
            self.embeddings = tf.get_variable('embeddings')
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
