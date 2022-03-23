import tensorflow as tf
import numpy as np
import sys
from .sasrec import Encoder, LayerNormalization
from .modules import Encoder as Encoder2


class MultiHeadAttention_v2(tf.keras.layers.Layer):
    """
    This is a modified version of the original MHA with
    the query and key-value of different dimension. Specifically,
    the query is batch_size X sequence_length-1 and
    key/values are batch_size X sequence_length-1 X sequence_length-2
    """

    def __init__(self, attention_dim, num_heads, history_len, embeddings, final_dim):
        super(MultiHeadAttention_v2, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.history_len = history_len
        assert attention_dim % self.num_heads == 0
        self.depth = attention_dim // self.num_heads
        self.embeddings = embeddings

        self.Q = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.K = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.V = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.final = tf.keras.layers.Dense(final_dim, activation=None)

    def call(self, queries, keys):
        # queries (current items), (None, s1)
        # keys, values (item history), (None, s1, s2)
        queries = self.embeddings(queries)  # (None, s1, d)
        keys = self.embeddings(keys)  # (None, s1, s2, d)
        # print(queries.shape, keys.shape)

        # Linear projections
        Q = self.Q(queries)  # (b, s1, d)
        K = self.K(keys)  # (b, s1, s2, d)
        V = self.V(keys)  # (b, s1, s2, d)
        # print(Q.shape, K.shape, V.shape)
        # print(self.num_heads, self.history_len)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (b*N, s1, d/h)
        K_ = tf.concat(
            tf.split(K, self.num_heads, axis=3), axis=0
        )  # (b*N, s1, s2, d/h)
        V_ = tf.concat(
            tf.split(V, self.num_heads, axis=3), axis=0
        )  # (b*N, s1, s2, d/h)
        # print(Q_.shape, K_.shape, V_.shape)

        # print(Q_.shape, K_.shape, V_.shape)
        Q_ = tf.expand_dims(Q_, axis=-2)  # (None, s1, 1, h)
        # Q_ = tf.tile(Q_, [1, 1, self.history_len, 1])
        outputs = tf.linalg.matmul(Q_, tf.transpose(K_, [0, 1, 3, 2]))
        # print(outputs.shape)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        # (None, s1, s2, s2)
        # print(outputs.shape)

        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (None, s1, s2)
        # print(key_masks.shape)
        key_masks = tf.tile(key_masks, [self.num_heads, 1, 1])  # (None, s1, s2)
        # print(key_masks.shape)
        key_masks = tf.tile(
            tf.expand_dims(key_masks, 2),
            [1, 1, 1, 1]
            # tf.expand_dims(key_masks, 2), [1, 1, self.history_len, 1]
        )  # (h*N, T_q, T_k)
        # print(key_masks.shape)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs, axis=-1)  # (None, s1, s2, s2)
        outputs = tf.matmul(outputs, V_)  # (None, s1, s2, h)
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=3)
        # (None, 50, 1, 100)
        # print(outputs.shape, queries.shape)
        outputs = tf.squeeze(outputs, axis=-2)  # (None, 50, 100)

        # Residual connection
        outputs += queries

        # to match with the other dimensions
        outputs = self.final(outputs)

        return outputs


class MultiHeadAttention4D(tf.keras.layers.Layer):
    """
    - Q (query), K (key) and V (value) are split into multiple heads (num_heads)
    - each tuple (q, k, v) are fed to scaled_dot_product_attention
    - all attention outputs are concatenated
    """

    def __init__(self, attention_dim, num_heads, dropout_rate):
        super(MultiHeadAttention4D, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        assert attention_dim % self.num_heads == 0
        self.dropout_rate = dropout_rate

        self.depth = attention_dim // self.num_heads

        self.Q = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.K = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.V = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, queries, keys):

        # Linear projections
        Q = self.Q(queries)  # (N, T_q, C)
        K = self.K(keys)  # (N, T_k, C)
        V = self.V(keys)  # (N, T_k, C)

        # --- MULTI HEAD ---
        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=3), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=3), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=3), axis=0)  # (h*N, T_k, C/h)

        # --- SCALED DOT PRODUCT ---
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 1, 3, 2]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # print("outputs", outputs.shape)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)

        # print("1.", key_masks.shape, self.num_heads)

        key_masks = tf.tile(key_masks, [self.num_heads, 1, 1])  # (h*N, T_k)

        # print("2.", key_masks.shape)

        key_masks = tf.tile(
            tf.expand_dims(key_masks, 2), [1, 1, tf.shape(queries)[2], 1]
        )  # (h*N, T_q, T_k)

        # print("3.", key_masks.shape)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Future blinding (Causality)
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(
            diag_vals
        ).to_dense()  # (T_q, T_k)

        # print("4.", tril.shape, outputs.shape)

        masks = tf.tile(
            tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1, 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)

        # print("5.", query_masks.shape)
        query_masks = tf.tile(query_masks, [self.num_heads, 1, 1])  # (h*N, T_q)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, 1, tf.shape(keys)[2]]
        )  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # print("6.", outputs.shape, queries.shape)

        # --- MULTI HEAD ---
        # concat heads
        outputs = tf.concat(
            tf.split(outputs, self.num_heads, axis=0), axis=3
        )  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # print("7.", outputs.shape)
        return outputs


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


class EncoderLayer4D(tf.keras.layers.Layer):
    """
    Transformer based encoder layer

    """

    def __init__(
        self,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        super(EncoderLayer4D, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        self.mha = MultiHeadAttention4D(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(conv_dims, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def call(self, x):

        x_norm = self.layer_normalization(x)
        attn_output = self.mha(queries=x_norm, keys=x)
        attn_output = self.ffn(attn_output)
        out = attn_output

        return out


class Encoder4D(tf.keras.layers.Layer):
    """
    Invokes Transformer based encoder with user defined number of layers

    """

    def __init__(
        self,
        num_layers,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        super(Encoder4D, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer4D(
                seq_max_len,
                embedding_dim,
                attention_dim,
                num_heads,
                conv_dims,
                dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)


class HSASREC(tf.keras.Model):
    """
    Hierarchical SASREc

    This model applies Transformer at two levels. The first one
    encodes the previous user history for each item and the
    second one encodes the item history for each user.

    """

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        super(HSASREC, self).__init__()
        self.user_num = kwargs.get("user_num", None)  # New
        self.user_len = kwargs.get("user_len", 50)  # New

        self.item_num = kwargs.get("item_num", None)
        self.seq_max_len = kwargs.get("seq_max_len", 100)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.attention_dim = kwargs.get("attention_dim", 100)
        self.attention_num_heads = kwargs.get("attention_num_heads", 1)
        # self.conv_dims = kwargs.get("conv_dims", [200, 200])
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.num_neg_test = kwargs.get("num_neg_test", 100)

        self.user_embedding_dim = kwargs.get("user_embedding_dim", self.embedding_dim)
        self.history_embedding_dim = kwargs.get(
            "history_embedding_dim", self.embedding_dim
        )
        self.history_attention_dim = kwargs.get(
            "history_attention_dim", self.attention_dim
        )

        # if user embedding is also used
        self.hidden_units = self.embedding_dim + self.user_embedding_dim
        # self.hidden_units = self.embedding_dim + self.history_embedding_dim

        self.conv_dims = [self.hidden_units, self.hidden_units]

        # if no user embedding is used
        # self.hidden_units = self.embedding_dim
        self.use_item_history = True  # for ablation study
        self.use_user_history = True  # for ablation study

        # User embedding & encoding
        self.user_embedding_layer = tf.keras.layers.Embedding(
            self.user_num + 1,
            self.user_embedding_dim,
            name="user_embeddings",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        if self.use_item_history:

            # self.user_encoder = Encoder4D(
            #     self.num_blocks,
            #     self.seq_max_len,
            #     self.user_embedding_dim,
            #     self.user_attention_dim,
            #     self.attention_num_heads,
            #     self.conv_dims,
            #     self.dropout_rate,
            # )
            # self.user_encoder = Encoder2(
            #     self.num_blocks,  # num_layers,
            #     self.user_embedding_dim,  # d_model
            #     self.attention_num_heads,  # num_heads
            #     self.seq_max_len,  # seq_len
            #     self.user_len,  # seq_len2
            #     self.user_attention_dim,
            #     # self.conv_dims,
            #     self.dropout_rate,
            # )
            self.history_encoder = tf.keras.layers.GRU(
                units=self.hidden_units, return_sequences=True, return_state=False
            )

            self.mlp_history = tf.keras.layers.Dense(
                units=self.hidden_units, activation="relu"
            )

        # item embedding
        self.item_embedding_layer = tf.keras.layers.Embedding(
            self.item_num + 1,
            self.embedding_dim,
            name="item_embeddings",
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        # item position embedding
        self.positional_embedding_layer = tf.keras.layers.Embedding(
            self.seq_max_len,
            # self.embedding_dim,
            self.hidden_units,
            name="item_user_positional_embeddings",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.hidden_units,
            self.hidden_units,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.hidden_units, 1e-08
        )
        self.attention_encoder = MultiHeadAttention_v2(
            self.attention_dim,
            self.attention_num_heads,
            self.user_len,
            self.item_embedding_layer,
            self.hidden_units,
        )

    def history_embedding(self, input_seq, training):
        """
        Encodes past user history for each item
        input_seq = (None, item_seq_len, user_seq_len)
        seq_embeddings = (None, s1, s2, d)
        """

        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (self.embedding_dim ** 0.5)

        positional_seq = tf.expand_dims(
            tf.expand_dims(tf.range(tf.shape(input_seq)[2]), 0), 0
        )
        positional_seq = tf.tile(
            positional_seq, [tf.shape(input_seq)[0], tf.shape(input_seq)[1], 1]
        )
        positional_embeddings = self.positional_embedding_layer(positional_seq)
        seq_embeddings += positional_embeddings  # (None, 50, 10, 100)

        # RNN Based Encoder (cannot take 4-dimensional input)
        seq_embeddings = tf.reshape(
            seq_embeddings,
            [-1, self.seq_max_len, self.user_len * self.history_embedding_dim],
        )  # (None, 50, 1000)
        hist_attention = self.history_encoder(seq_embeddings)  # (None, 50, 200)

        # Transformer Based Encoder (takes 4-dimensional input)
        # user_attention = self.user_encoder(seq_embeddings, training, mask=None)
        # merge over all the user embeddings
        # user_attention = tf.reshape(
        #     user_attention,
        #     [-1, self.seq_max_len, self.user_len * self.user_embedding_dim],
        # )
        hist_attention = self.mlp_history(hist_attention)
        hist_attention = hist_attention * (self.hidden_units ** 0.5)

        return hist_attention

    def embedding(self, input_seq):

        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (
            self.embedding_dim ** 0.5
        )  # should be added?

        # FIXME
        positional_seq = tf.expand_dims(tf.range(self.seq_max_len), 0)
        positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
        positional_embeddings = self.positional_embedding_layer(positional_seq)

        return seq_embeddings, positional_embeddings

    def call(self, x, training):

        users = x["users"]
        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]
        his = x["item_history"]

        # user-embedding for the item history
        u_latent = self.user_embedding_layer(users)
        u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)
        # replicate the user embedding for all the items
        u_latent = tf.tile(u_latent, [1, self.seq_max_len, 1])  # (b, s, h)

        # item embeddings
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        seq_embeddings = tf.reshape(
            tf.concat([seq_embeddings, u_latent], 2),
            [-1, self.seq_max_len, self.hidden_units],
        )

        # add positional embeddings
        seq_embeddings += positional_embeddings

        # add user/history embeddings - dimension should be
        # (None, seq_len, embedding_dim)
        if self.use_item_history:
            hist_attention = self.attention_encoder(input_seq, his)
            # hist_attention = self.history_embedding(his, training)
            seq_embeddings += hist_attention

        # dropout
        seq_embeddings = self.dropout_layer(seq_embeddings, training=training)

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

        user_emb = tf.reshape(
            u_latent,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.user_embedding_dim],
        )
        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)

        # Add user embeddings
        pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units])
        neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units])

        # print(pos_emb.shape, neg_emb.shape, seq_attention.shape)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.hidden_units],
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
        user = x["user"]
        input_seq = x["input_seq"]
        candidate = x["candidate"]
        his = x["item_history"]

        u0_latent = self.user_embedding_layer(user)
        u0_latent = u0_latent * (self.user_embedding_dim ** 0.5)  # (1, 1, h)
        u0_latent = tf.squeeze(u0_latent, axis=0)  # (1, h)
        test_user_emb = tf.tile(u0_latent, [1 + self.num_neg_test, 1])  # (101, h)

        u_latent = self.user_embedding_layer(user)
        u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)
        u_latent = tf.tile(u_latent, [1, tf.shape(input_seq)[1], 1])  # (b, s, h)

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)
        seq_embeddings = tf.reshape(
            tf.concat([seq_embeddings, u_latent], 2),
            [tf.shape(input_seq)[0], -1, self.hidden_units],
        )
        seq_embeddings += positional_embeddings  # (b, s, h1 + h2)

        if self.use_item_history:
            # user history embeddings
            hist_attention = self.attention_encoder(input_seq, his)
            # hist_attention = self.history_embedding(his, training)
            seq_embeddings += hist_attention

        # seq_embeddings = self.dropout_layer(seq_embeddings)
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.hidden_units],
        )  # (b*s, d)

        candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        candidate_emb = tf.squeeze(candidate_emb, axis=0)  # (s2, h2)
        candidate_emb = tf.reshape(
            tf.concat([candidate_emb, test_user_emb], 1), [-1, self.hidden_units]
        )  # (b*s2, h1+h2)
        candidate_emb = tf.transpose(candidate_emb, perm=[1, 0])  # (h1+h2, b*s2)
        # candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)
        # print(seq_emb.shape, candidate_emb.shape)

        test_logits = tf.matmul(seq_emb, candidate_emb)  # (200, 100) * (1, 101, 100)'
        # print(test_logits.shape)

        test_logits = tf.reshape(
            test_logits,
            [tf.shape(input_seq)[0], self.seq_max_len, 1 + self.num_neg_test],
        )  # (1, 200, 101)
        test_logits = test_logits[:, -1, :]  # (1, 101)
        return test_logits
