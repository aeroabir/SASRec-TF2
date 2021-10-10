import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys

from .sasrec import LayerNormalization


class PointWiseFeedForward(tf.keras.layers.Layer):
    """
    Convolution layers with residual connection
    """

    def __init__(self, conv_dims, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv_dims = conv_dims
        self.dropout_rate = dropout_rate
        self.conv_layer1 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[0], kernel_size=1, activation="relu", use_bias=True
        )
        self.conv_layer2 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[1], kernel_size=1, activation=None, use_bias=True
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):

        output = self.conv_layer1(x)
        output = self.dropout_layer(output)

        output = self.conv_layer2(output)
        output = self.dropout_layer(output)

        # Residual connection
        output += x

        return output


class EncoderLayer(tf.keras.layers.Layer):
    """
    Transformer based encoder layer

    """

    def __init__(
        self,
        seq_max_len,
        embedding_dim,
        hidden_dim,
        rnn_name,
        conv_dims,
        dropout_rate,
    ):
        super(EncoderLayer, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        if rnn_name == "lstm":
            self.rnn = layers.LSTM(hidden_dim, return_sequences=True)
        else:
            self.rnn = layers.GRU(hidden_dim, return_sequences=True)
        # self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(conv_dims, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def call_(self, x, training, mask):

        attn_output = self.mha(queries=self.layer_normalization(x), keys=x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # feed forward network
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        # masking
        out2 *= mask

        return out2

    def call(self, x, training, mask):

        x_norm = self.layer_normalization(x)
        attn_output = self.rnn(x_norm)
        attn_output = self.ffn(attn_output)
        out = attn_output * mask

        return out


class Encoder(tf.keras.layers.Layer):
    """
    Invokes RNN based encoder with user defined number of layers

    """

    def __init__(
        self,
        num_layers,
        seq_max_len,
        embedding_dim,
        hidden_dim,
        rnn_name,
        conv_dims,
        dropout_rate,
    ):
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(
                seq_max_len,
                embedding_dim,
                hidden_dim,
                rnn_name,
                conv_dims,
                dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class RNNREC(tf.keras.Model):
    """RNN Rec model
    RNN Based Sequential Prediction

    Args:
        item_num: number of items in the dataset
        seq_max_len: maximum number of items in user history
        num_blocks: number of RNN blocks to be stacked
        embedding_dim: item embedding dimension
        hidden_dim: RNN hidden dimension
        rnn_name: RNN type, LSTM or GRU
        conv_dims: list of the dimensions of the Feedforward layer
        dropout_rate: dropout rate
        l2_reg: coefficient of the L2 regularization
        num_neg_test: number of negative examples used in testing
    """

    def __init__(self, **kwargs):
        super(RNNREC, self).__init__()

        self.item_num = kwargs.get("item_num", None)
        self.seq_max_len = kwargs.get("seq_max_len", 100)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.hidden_dim = kwargs.get("hidden_dim", 100)
        self.rnn_name = kwargs.get("rnn_name", "gru")
        self.conv_dims = kwargs.get("conv_dims", [100, 100])
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.num_neg_test = kwargs.get("num_neg_test", 100)

        self.item_embedding_layer = tf.keras.layers.Embedding(
            self.item_num + 1,
            self.embedding_dim,
            name="item_embeddings",
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        # if self.rnn_name == "lstm":
        #     self.rnn = layers.LSTM(self.hidden_dim, return_sequences=True)
        # else:
        #     self.rnn = layers.GRU(self.hidden_dim, return_sequences=True)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.embedding_dim,
            self.hidden_dim,
            self.rnn_name,
            self.conv_dims,
            self.dropout_rate,
        )
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def call(self, x, training):

        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (self.embedding_dim ** 0.5)

        # dropout
        seq_embeddings = self.dropout_layer(seq_embeddings)

        # masking
        seq_embeddings *= mask

        seq_attention = seq_embeddings
        # seq_attention = self.rnn(seq_attention)
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)

        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)

        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        pos_logits = tf.expand_dims(pos_logits, axis=-1)  # (bs, 1)
        neg_logits = tf.expand_dims(neg_logits, axis=-1)  # (bs, 1)

        # masking for loss calculation
        istarget = tf.reshape(
            tf.cast(tf.not_equal(pos, 0), dtype=tf.float32),
            [tf.shape(input_seq)[0] * self.seq_max_len],
        )

        return pos_logits, neg_logits, istarget

    def predict(self, x):
        training = False
        input_seq = x["input_seq"]
        candidate = x["candidate"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (self.embedding_dim ** 0.5)
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        # seq_attention = self.rnn(seq_attention)
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)
        candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)
        # print(seq_emb.shape, candidate_emb.shape)

        test_logits = tf.matmul(seq_emb, candidate_emb)  # (200, 100) * (1, 101, 100)'
        # print(test_logits.shape)

        test_logits = tf.reshape(
            test_logits,
            [tf.shape(input_seq)[0], self.seq_max_len, 1 + self.num_neg_test],
        )  # (1, 200, 101)
        test_logits = test_logits[:, -1, :]  # (1, 101)
        return test_logits
