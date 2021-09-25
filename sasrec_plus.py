import tensorflow as tf
import numpy as np
import sys
from sasrec import SASREC


class TextEncoder(tf.keras.layers.Layer):
    """
    Text encoder is an LSTM that takes sequence of word vectors
    as input and returns the final state

    """

    def __init__(self, embedding, out_dim, dropout_rate):
        super(TextEncoder, self).__init__()
        self.embedding = embedding
        # initializers, 'glorot_uniform', 'he_normal' both cause NaN
        self.lstm_layer = tf.keras.layers.LSTM(
            units=out_dim,
            return_sequences=True,
            return_state=False,
            recurrent_initializer="he_normal",
        )
        # self.linear = tf.keras.layers.Dense(units=out_dim,
        #                                     activation='relu',
        #                                     kernel_initializer='he_normal')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        y = self.embedding(x)  # ([b, s, h2])
        # y = tf.reduce_mean(y, 2)  # [b, s, h2], average over the text length
        y = self.lstm_layer(y)  # [b, s, h]
        # y = self.linear(y)  # [b, s, h2]
        y = self.dropout(y, training=training)
        return y


class SASREC_PLUS(SASREC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.extra_embedding_matrix = kwargs.get("item_text_embedding_matrix", None)

        self.extra_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.item_num + 1,
            output_dim=self.extra_embedding_matrix.shape[1],
            name="extra_embeddings",
            weights=[self.extra_embedding_matrix],
            #   input_length=self.max_seq_len_text,
            trainable=False,
        )

        self.text_encoder = TextEncoder(
            self.extra_embedding_layer, self.embedding_dim, self.dropout_rate
        )

    def embedding(self, input_seq):

        seq_embeddings = self.item_embedding_layer(input_seq)
        text_embeddings = self.text_encoder(input_seq)  # [b, s, 100]

        # add text embeddings
        seq_embeddings += text_embeddings

        seq_embeddings = seq_embeddings * (
            self.embedding_dim ** 0.5
        )  # should be added?

        # FIXME
        positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
        positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
        positional_embeddings = self.positional_embedding_layer(positional_seq)

        return seq_embeddings, positional_embeddings

    def call(self, x, training):

        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(
            input_seq
        )  # [b, 50, 100]

        # add positional embeddings
        seq_embeddings += positional_embeddings

        # dropout
        seq_embeddings = self.dropout_layer(seq_embeddings)

        # masking
        seq_embeddings *= mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = self.mask_layer(pos)
        neg = self.mask_layer(neg)

        # pos_tokens = self.item_text_sequences[pos,:]  # (b, s, s2)
        # pos_tokens = x['pos_tokens']
        # pos_text_embeddings = self.text_encoder(pos_tokens)  # [b, s, 100]
        pos_text_embeddings = self.text_encoder(pos)  # [b, s, 100]
        pos_text_embeddings = tf.reshape(
            pos_text_embeddings, [tf.shape(input_seq)[0] * self.seq_max_len, -1]
        )

        # neg_tokens = self.item_text_sequences[neg,:]  # (b, s, s2)
        # neg_tokens = x['neg_tokens']
        neg_text_embeddings = self.text_encoder(neg)  # [b, s, 100]
        neg_text_embeddings = tf.reshape(
            neg_text_embeddings, [tf.shape(input_seq)[0] * self.seq_max_len, -1]
        )

        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)

        pos_emb += pos_text_embeddings
        neg_emb += neg_text_embeddings

        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)

        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        pos_logits = tf.expand_dims(pos_logits, axis=-1)  # (bs, 1)
        # pos_prob = tf.keras.layers.Dense(1, activation='sigmoid')(pos_logits)  # (bs, 1)

        neg_logits = tf.expand_dims(neg_logits, axis=-1)  # (bs, 1)
        # neg_prob = tf.keras.layers.Dense(1, activation='sigmoid')(neg_logits)  # (bs, 1)

        # output = tf.concat([pos_logits, neg_logits], axis=0)

        # masking for loss calculation
        istarget = tf.reshape(
            tf.cast(tf.not_equal(pos, 0), dtype=tf.float32),
            [tf.shape(input_seq)[0] * self.seq_max_len],
        )

        # return input_seq_tokens, text_embeddings  # test
        # return pos_tokens, pos_text_embeddings  # test
        # return pos_emb, seq_emb  # test
        return pos_logits, neg_logits, istarget

    def predict(self, x):
        training = False
        input_seq = x["input_seq"]
        candidate = x["candidate"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        # text embedding is included here
        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # text_embeddings = self.text_encoder(input_seq)  # [b, s, 100]
        # seq_embeddings += text_embeddings
        seq_embeddings += positional_embeddings  # (1, 50, 100)

        # seq_embeddings = self.dropout_layer(seq_embeddings)
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)

        candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        candidate_text_embeddings = self.text_encoder(candidate)  # [b, s, 100]
        # candidate_text_embeddings = tf.reshape(candidate_text_embeddings, [tf.shape(input_seq)[0] * self.seq_max_len, -1])
        candidate_emb += candidate_text_embeddings  # (1, 101, 100)

        candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)
        # print(seq_emb.shape, candidate_emb.shape)

        test_logits = tf.matmul(seq_emb, candidate_emb)  # (200, 100) * (1, 101, 100)'

        test_logits = tf.reshape(
            test_logits,
            [tf.shape(input_seq)[0], self.seq_max_len, 1 + self.num_neg_test],
        )  # (1, 200, 101)
        test_logits = test_logits[:, -1, :]  # (1, 101)
        return test_logits
