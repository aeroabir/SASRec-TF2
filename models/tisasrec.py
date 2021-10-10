import tensorflow as tf
import numpy as np
import sys


class MultiHeadAttention(tf.keras.layers.Layer):
    """

    :Citation:

    Peter Shaw, Jakob Uszkoreit and Ashish Vaswani (2018),
    Self-Attention with Relative Position Representations

    - Q (query), K (key) and V (value) are split into multiple heads
    - each tuple (q, k, v) are fed to scaled_dot_product_attention
    - all attention outputs are concatenated

    - there are separate position embeddings and embeddings that
    take into account of the distance between sequence elements

    """

    def __init__(self, attention_dim, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        assert attention_dim % self.num_heads == 0
        self.dropout_rate = dropout_rate

        self.depth = attention_dim // self.num_heads

        self.Q = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.K = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.V = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(
        self,
        queries,
        keys,
        time_matrix_K,
        time_matrix_V,
        absolute_pos_K,
        absolute_pos_V,
    ):

        # Linear projections
        Q = self.Q(queries)  # (N, T_q, C)
        K = self.K(keys)  # (N, T_k, C)
        V = self.V(keys)  # (N, T_k, C)

        # --- MULTI HEAD ---
        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        time_matrix_K_ = tf.concat(
            tf.split(time_matrix_K, self.num_heads, axis=3), axis=0
        )
        time_matrix_V_ = tf.concat(
            tf.split(time_matrix_V, self.num_heads, axis=3), axis=0
        )
        absolute_pos_K_ = tf.concat(
            tf.split(absolute_pos_K, self.num_heads, axis=2), axis=0
        )
        absolute_pos_V_ = tf.concat(
            tf.split(absolute_pos_V, self.num_heads, axis=2), axis=0
        )

        # --- SCALED DOT PRODUCT ---
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        # print("0. outputs:", outputs.shape)

        outputs_pos = tf.matmul(Q_, tf.transpose(absolute_pos_K_, [0, 2, 1]))
        # time_matrix_K_.shape, Q_.shape = (None, 50, 50, 100), (None, 50, 100)
        Q__ = tf.expand_dims(Q_, axis=-1)
        outputs_time = tf.matmul(time_matrix_K_, Q__)
        outputs_time = tf.squeeze(outputs_time, -1)
        # outputs_time = tf.squeeze(
        #     tf.matmul(time_matrix_K_, tf.expand_dims(Q_, axis=-1))
        # )
        # print("1. outputs:", outputs_time.shape)

        outputs = outputs + outputs_time + outputs_pos
        # print("2. outputs:", outputs.shape, outputs_time.shape, outputs_pos.shape)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(
            tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Future blinding (Causality)
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(
            diag_vals
        ).to_dense()  # (T_q, T_k)
        masks = tf.tile(
            tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]
        )  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs_value = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        outputs_pos_value = tf.matmul(outputs, absolute_pos_V_)
        output_time_value = tf.reshape(
            tf.matmul(tf.expand_dims(outputs, axis=2), time_matrix_V_),
            [tf.shape(outputs_pos)[0], tf.shape(outputs_pos)[1], self.attention_dim],
        )

        outputs = outputs_value + output_time_value + outputs_pos_value

        # --- MULTI HEAD ---
        # concat heads
        outputs = tf.concat(
            tf.split(outputs, self.num_heads, axis=0), axis=2
        )  # (N, T_q, C)

        # Residual connection
        outputs += queries

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
            filters=self.conv_dims[0],
            kernel_size=1,
            activation="relu",
            use_bias=True,
            # input_shape=(200, 100),
        )
        self.conv_layer2 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[1],
            kernel_size=1,
            activation=None,
            use_bias=True,
            # input_shape=(200, 100),
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
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        super(EncoderLayer, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(conv_dims, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def call_(
        self,
        x,
        time_matrix_emb_K,
        time_matrix_emb_V,
        absolute_pos_K,
        absolute_pos_V,
        training,
        mask,
    ):

        attn_output = self.mha(
            queries=self.layer_normalization(x),
            keys=x,
            time_matrix_K=time_matrix_emb_K,
            time_matrix_V=time_matrix_emb_V,
            absolute_pos_K=absolute_pos_K,
            absolute_pos_V=absolute_pos_V,
        )
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

    def call(
        self,
        x,
        time_matrix_emb_K,
        time_matrix_emb_V,
        absolute_pos_K,
        absolute_pos_V,
        training,
        mask,
    ):

        x_norm = self.layer_normalization(x)
        attn_output = self.mha(
            queries=x_norm,
            keys=x,
            time_matrix_K=time_matrix_emb_K,
            time_matrix_V=time_matrix_emb_V,
            absolute_pos_K=absolute_pos_K,
            absolute_pos_V=absolute_pos_V,
        )
        attn_output = self.ffn(attn_output)
        out = attn_output * mask

        return out


class Encoder(tf.keras.layers.Layer):
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
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(
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

    def call(
        self,
        x,
        tm_emb_K,
        tm_emb_V,
        pos_K,
        pos_V,
        training,
        mask,
    ):

        time_matrix_emb_K = tm_emb_K
        time_matrix_emb_V = tm_emb_V
        absolute_pos_K = pos_K
        absolute_pos_V = pos_V
        # it is important to check the rank of the resulting
        # matrices and there should be only one None that
        # corresponds to the batch size

        for i in range(self.num_layers):
            # print("begin-x:", x.shape)
            x = self.enc_layers[i](
                x,
                time_matrix_emb_K,
                time_matrix_emb_V,
                absolute_pos_K,
                absolute_pos_V,
                training,
                mask,
            )
            # print("after-x:", x.shape)

        return x  # (batch_size, input_seq_len, d_model)


class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization using mean and variance
    gamma and beta are the learnable parameters
    """

    def __init__(self, seq_max_len, embedding_dim, epsilon):
        super(LayerNormalization, self).__init__()
        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.params_shape = (self.seq_max_len, self.embedding_dim)
        g_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=g_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=b_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, [-1], keepdims=True)
        normalized = (x - mean) / ((variance + self.epsilon) ** 0.5)
        output = self.gamma * normalized + self.beta
        return output


class TISASREC(tf.keras.Model):
    """TiSASRec model
    Self-Attentive Sequential Recommendation Using Time-Interval Aware Transformer

    :Citation:

        Jiacheng Li, Yujie Wang and Julian McAuley (2020),
        Time Interval Aware Self-Attention for Sequential Recommendation (WSDM'20)

        Original source code in TF1.x: https://github.com/JiachengLi1995/TiSASRec

    Args:
        item_num: number of items in the dataset
        seq_max_len: maximum number of items in user history
        num_blocks: number of Transformer blocks to be used
        embedding_dim: item embedding dimension
        attention_dim: Transformer attention dimension
        conv_dims: list of the dimensions of the Feedforward layer
        dropout_rate: dropout rate
        l2_reg: coefficient of the L2 regularization
        num_neg_test: number of negative examples used in testing
    """

    def __init__(self, **kwargs):
        super(TISASREC, self).__init__()

        self.item_num = kwargs.get("item_num", None)
        self.seq_max_len = kwargs.get("seq_max_len", 100)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.attention_dim = kwargs.get("attention_dim", 100)
        self.attention_num_heads = kwargs.get("attention_num_heads", 1)
        self.conv_dims = kwargs.get("conv_dims", [100, 100])
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.num_neg_test = kwargs.get("num_neg_test", 100)
        self.time_span = kwargs.get("time_span", 256)

        self.item_embedding_layer = tf.keras.layers.Embedding(
            self.item_num + 1,
            self.embedding_dim,
            name="item_embeddings",
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        # position embeddings for key
        self.positional_embedding_layer_K = tf.keras.layers.Embedding(
            self.seq_max_len,
            self.embedding_dim,
            name="positional_embeddings_K",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        # position embeddings for value
        self.positional_embedding_layer_V = tf.keras.layers.Embedding(
            self.seq_max_len,
            self.embedding_dim,
            name="positional_embeddings_V",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        # embedding for time interval (key)
        self.time_matrix_embedding_layer_K = tf.keras.layers.Embedding(
            input_dim=self.time_span + 1,
            output_dim=self.embedding_dim,
            name="time_embeddings_K",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        # embedding for time interval (value)
        self.time_matrix_embedding_layer_V = tf.keras.layers.Embedding(
            input_dim=self.time_span + 1,
            output_dim=self.embedding_dim,
            name="time_embeddings_V",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.embedding_dim,
            self.attention_dim,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def embedding(self, input_seq):

        # input_seq is (None, seq_len)
        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (
            self.embedding_dim ** 0.5
        )  # should be added?

        positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
        positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])

        positional_embeddings_K = self.positional_embedding_layer_K(positional_seq)
        positional_embeddings_V = self.positional_embedding_layer_V(positional_seq)

        return seq_embeddings, positional_embeddings_K, positional_embeddings_V

    def call(self, x, training):

        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]
        tm = x["time_matrix"]
        # pos = x["position"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        (
            seq_embeddings,
            absolute_pos_K,
            absolute_pos_V,
        ) = self.embedding(input_seq)

        time_matrix_emb_K = self.time_matrix_embedding_layer_K(tm)
        time_matrix_emb_V = self.time_matrix_embedding_layer_V(tm)

        # dropout & masking
        seq_embeddings = self.dropout_layer(seq_embeddings, training=training)
        seq_embeddings *= mask

        # dropout & masking
        time_matrix_emb_K = self.dropout_layer(time_matrix_emb_K, training=training)
        time_matrix_emb_V = self.dropout_layer(time_matrix_emb_V, training=training)
        absolute_pos_K = self.dropout_layer(absolute_pos_K, training=training)
        absolute_pos_V = self.dropout_layer(absolute_pos_V, training=training)

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings  # (b, seqlen, hidden_dim)
        seq_attention = self.encoder(
            seq_attention,
            time_matrix_emb_K,
            time_matrix_emb_V,
            absolute_pos_K,
            absolute_pos_V,
            training,
            mask,
        )  # (b, seqlen, hidden_dim)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = self.mask_layer(pos)
        neg = self.mask_layer(neg)

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
        # pos_prob = tf.keras.layers.Dense(1, activation='sigmoid')(pos_logits)  # (bs, 1)

        neg_logits = tf.expand_dims(neg_logits, axis=-1)  # (bs, 1)
        # neg_prob = tf.keras.layers.Dense(1, activation='sigmoid')(neg_logits)  # (bs, 1)

        # output = tf.concat([pos_logits, neg_logits], axis=0)

        # masking for loss calculation
        istarget = tf.reshape(
            tf.cast(tf.not_equal(pos, 0), dtype=tf.float32),
            [tf.shape(input_seq)[0] * self.seq_max_len],
        )

        return pos_logits, neg_logits, istarget

    def predict(self, inputs):
        training = False
        input_seq = inputs["input_seq"]
        candidate = inputs["candidate"]
        tm = inputs["time_matrix"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        (
            seq_embeddings,
            absolute_pos_K,
            absolute_pos_V,
        ) = self.embedding(input_seq)
        time_matrix_emb_K = self.time_matrix_embedding_layer_K(tm)
        time_matrix_emb_V = self.time_matrix_embedding_layer_V(tm)

        # dropout & masking
        seq_embeddings = self.dropout_layer(seq_embeddings, training=training)
        seq_embeddings *= mask

        time_matrix_emb_K = self.dropout_layer(time_matrix_emb_K, training=training)
        time_matrix_emb_V = self.dropout_layer(time_matrix_emb_V, training=training)
        absolute_pos_K = self.dropout_layer(absolute_pos_K, training=training)
        absolute_pos_V = self.dropout_layer(absolute_pos_V, training=training)

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings  # (b, seqlen, hidden_dim)
        seq_attention = self.encoder(
            seq_attention,
            time_matrix_emb_K,
            time_matrix_emb_V,
            absolute_pos_K,
            absolute_pos_V,
            training,
            mask,
        )  # (b, seqlen, hidden_dim)
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

    def loss_function(self, pos_logits, neg_logits):

        # loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # pos_labels = np.repeat(1, pos_logits.shape[0])
        # pos_labels = np.expand_dims(pos_labels, axis=-1)
        # pos_loss = loss_fn(pos_labels, pos_logits)

        # neg_labels = np.repeat(0, neg_logits.shape[0])
        # neg_labels = np.expand_dims(neg_labels, axis=-1)
        # neg_loss = loss_fn(neg_labels, neg_logits)

        # # print(pos_loss, neg_loss)
        # loss = pos_loss + neg_loss

        # ignore padding items (0)
        istarget = tf.reshape(
            tf.cast(tf.not_equal(self.pos, 0), dtype=tf.float32),
            [tf.shape(self.input_seq)[0] * self.seq_max_len],
        )
        loss = tf.reduce_sum(
            -tf.math.log(pos_logits + 1e-24) * istarget
            - tf.math.log(1 - neg_logits + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        # loss += sum(reg_losses)
        loss += reg_loss

        return loss
