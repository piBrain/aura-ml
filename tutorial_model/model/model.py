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

    def encode(self, num_units, seq_len, cell=None):
        if cell:
            encoder_cell = cell
        else:
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            encoder_cell,
            encoder_cell,
            self.enc_embedding,
            sequence_length=seq_len,
            time_major=self.time_major,
            dtype=tf.float32
        )

        return tf.concat(encoder_outputs, -1), encoder_state


    def decode(
        self, num_units, out_seq_len,
        encoder_state, cell=None, helper=None
    ):

        if cell:
            decoder_cell = cell
        else:
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

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
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=20,
            impute_finished=True
        )
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
            weights = tf.Variable(
                initial_value=tf.random_uniform(
                    [self.batch_size, out_seq_len],
                    dtype=tf.float32,
                    name='weight_uniform_initialiser'
                ),
                dtype=tf.float32,
                trainable=True
            )
            loss = tf.contrib.seq2seq.sequence_loss(
               t_out,
               labels,
               weights,
               average_across_batch=self.avg_across_batch,
               average_across_timesteps=self.avg_across_timestep
            )

        if not train_op:
            train_op = tf.layers.optimize_loss(
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
