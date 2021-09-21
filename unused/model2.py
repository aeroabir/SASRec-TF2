import tensorflow as tf
import numpy as np
import sys

class SASREC(tf.keras.Model):

    def __init__(self, **kwargs):
        super(SASREC, self).__init__()

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

        # self.model = self.build_model()


    def call(self, inputs):

        # return self.model(inputs)
        input_seq = inputs['input_seq']
        pos = inputs['positive']
        neg = inputs['negative']

        # for later usage
        self.input_seq = input_seq
        self.pos = pos
        self.neg = neg

        # FIXME 確認必要
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)

        # --- EMBEDDING LAYER ---
        self.build_embedding_layer()

        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # add positional embeddings
        seq_embeddings += positional_embeddings

        # dropout
        seq_embeddings = tf.keras.layers.Dropout(self.dropout_rate)(seq_embeddings)

        # masking
        seq_embeddings *= mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings

        for i in range(self.num_blocks):
            # dropoutとresidual connectionは各関数内で行う

            # attention layer
            seq_attention = self.multihead_attention(queries=self.layer_normalization(seq_attention),
                                                     keys=seq_attention)

            # feed forward network
            seq_attention = self.point_wise_feed_forward(seq_attention)

            # masking
            seq_attention *= mask

        # 実装ではここで再度layer_normalizationしているが、必要ない気がする
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = tf.keras.layers.Masking(mask_value=0)(pos)
        neg = tf.keras.layers.Masking(mask_value=0)(neg)

        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)
        seq_emb = tf.reshape(seq_attention, [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim]) # (b*s, d)

        # shared item embedding (for candidate item)
        # candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        # candidate_emb = tf.reshape(candidate_emb, [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim]) # (b*s, d)
        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        pos_logits = tf.expand_dims(pos_logits, axis=-1)  # (bs, 1)
        # pos_prob = tf.keras.layers.Dense(1, activation='sigmoid')(pos_logits)  # (bs, 1)

        neg_logits = tf.expand_dims(neg_logits, axis=-1)  # (bs, 1)
        # neg_prob = tf.keras.layers.Dense(1, activation='sigmoid')(neg_logits)  # (bs, 1)

        # 以下のように変更すれば他クラス問題に変更可能（のはず）
        # shared item embedding (for all items)
        # candidate_ids = tf.range(start=1, limit=item_num+1)
        # candidate_emb = item_embedding_layer(all_item_ids)

        # ユーザーの行動系列のembeddingとitemのembeddingを掛け合わせることでユーザーのimplicitな評価値を得る
        # output = tf.reduce_sum(candidate_emb * seq_emb, -1)  # (b*s, )
        # output = tf.expand_dims(output, axis=-1)  # (bs, 1)
        # output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
        output = tf.concat([pos_logits, neg_logits], axis=0)

        # return pos_logits, neg_logits
        return output


    def predict(self, inputs):
        input_seq = inputs['input_seq']
        candidate = inputs['candidate']

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)
        seq_embeddings += positional_embeddings
        seq_embeddings = tf.keras.layers.Dropout(self.dropout_rate)(seq_embeddings)
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        for i in range(self.num_blocks):
            seq_attention = self.multihead_attention(queries=self.layer_normalization(seq_attention),
                                                     keys=seq_attention)

            # feed forward network
            seq_attention = self.point_wise_feed_forward(seq_attention)

            # masking
            seq_attention *= mask
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        seq_emb = tf.reshape(seq_attention, [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim]) # (b*s, d)
        candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)
        # print(seq_emb.shape, candidate_emb.shape)

        test_logits = tf.matmul(seq_emb, candidate_emb)  # (200, 100) * (1, 101, 100)'
        # print(test_logits.shape)

        test_logits = tf.reshape(test_logits, [tf.shape(input_seq)[0], self.seq_max_len, 1+self.num_neg_test])  # (1, 200, 101)
        test_logits = test_logits[:, -1, :]  # (1, 101)
        return test_logits


    def loss_function(self, pos_prob, neg_prob):
        
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        pos_labels = np.repeat(1, pos_prob.shape[0])
        pos_labels = np.expand_dims(pos_labels, axis=-1)
        pos_loss = loss_fn(pos_labels, pos_prob)

        neg_labels = np.repeat(0, neg_prob.shape[0])
        neg_labels = np.expand_dims(neg_labels, axis=-1)
        neg_loss = loss_fn(neg_labels, neg_prob)

        # print(pos_loss, neg_loss)
        loss = pos_loss + neg_loss

        # ignore padding items (0)
        # istarget = tf.reshape(tf.cast(tf.not_equal(self.pos, 0), dtype=tf.float32), [tf.shape(self.input_seq)[0] * self.seq_max_len])
        # print(pos_prob, neg_prob)
        # loss = tf.reduce_sum(
        #      - tf.math.log(pos_logits + 1e-24) * istarget -
        #        tf.math.log(1 - neg_logits + 1e-24) * istarget
        # ) / tf.reduce_sum(istarget)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        # loss += sum(reg_losses)
        loss += reg_loss

        return loss


    def build_model(self):

        inputs = self.build_input()
        input_seq = inputs[0]
        pos = inputs[1]
        neg = inputs[2]

        # FIXME 確認必要
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)

        # --- EMBEDDING LAYER ---
        self.build_embedding_layer()

        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # add positional embeddings
        seq_embeddings += positional_embeddings

        # dropout
        seq_embeddings = tf.keras.layers.Dropout(self.dropout_rate)(seq_embeddings)

        # masking
        seq_embeddings *= mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings

        for i in range(self.num_blocks):
            # dropoutとresidual connectionは各関数内で行う

            # attention layer
            seq_attention = self.multihead_attention(queries=self.layer_normalization(seq_attention),
                                                     keys=seq_attention)

            # feed forward network
            seq_attention = self.point_wise_feed_forward(seq_attention)

            # masking
            seq_attention *= mask

        # 実装ではここで再度layer_normalizationしているが、必要ない気がする
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = tf.keras.layers.Masking(mask_value=0)(pos)
        neg = tf.keras.layers.Masking(mask_value=0)(neg)

        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)
        seq_emb = tf.reshape(seq_attention, [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim]) # (b*s, d)

        # shared item embedding (for candidate item)
        # candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        # candidate_emb = tf.reshape(candidate_emb, [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim]) # (b*s, d)
        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        pos_logits = tf.expand_dims(pos_logits, axis=-1)  # (bs, 1)
        # pos_prob = tf.keras.layers.Dense(1, activation='sigmoid')(pos_logits)  # (bs, 1)

        neg_logits = tf.expand_dims(neg_logits, axis=-1)  # (bs, 1)
        # neg_prob = tf.keras.layers.Dense(1, activation='sigmoid')(neg_logits)  # (bs, 1)

        # 以下のように変更すれば他クラス問題に変更可能（のはず）
        # shared item embedding (for all items)
        # candidate_ids = tf.range(start=1, limit=item_num+1)
        # candidate_emb = item_embedding_layer(all_item_ids)

        # ユーザーの行動系列のembeddingとitemのembeddingを掛け合わせることでユーザーのimplicitな評価値を得る
        # output = tf.reduce_sum(candidate_emb * seq_emb, -1)  # (b*s, )
        # output = tf.expand_dims(output, axis=-1)  # (bs, 1)
        # output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

        # model = tf.keras.Model(inputs=inputs, outputs=output)
        # model = tf.keras.Model(inputs=inputs, outputs=seq_embeddings)
        # model = tf.keras.Model(inputs=inputs, outputs=seq_attention)
        # model = tf.keras.Model(inputs=inputs, outputs=candidate_emb)
        model = tf.keras.Model(inputs=inputs, outputs=[pos_logits, neg_logits])
        # model = tf.keras.Model(inputs=inputs, outputs=[pos_prob, neg_prob])

        self.input_seq = input_seq
        self.pos = pos
        self.neg = neg

        return model


    def build_input(self):
        input_seq = tf.keras.layers.Input(shape=(self.seq_max_len,), name='input_seq')
        positive = tf.keras.layers.Input(shape=(self.seq_max_len,), name='positive')
        negative = tf.keras.layers.Input(shape=(self.seq_max_len,), name='negative')

        return [input_seq, positive, negative]


    def build_embedding_layer(self):

        item_embedding_layer = tf.keras.layers.Embedding(self.item_num,
                                                         self.embedding_dim,
                                                         name='item_embeddings',
                                                         mask_zero=True,
                                                         embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg))
        
        positional_embedding_layer = tf.keras.layers.Embedding(self.seq_max_len,
                                                               self.embedding_dim,
                                                               name='positional_embeddings',
                                                               mask_zero=False,
                                                               embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg))
        self.item_embedding_layer = item_embedding_layer
        self.positional_embedding_layer = positional_embedding_layer


    def embedding(self, input_seq):

        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (self.embedding_dim ** 0.5)  # should be added?

        # FIXME 確認が必要
        positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
        positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
        positional_embeddings = self.positional_embedding_layer(positional_seq)

        return seq_embeddings, positional_embeddings


    def layer_normalization(self, input_seq, epsilon = 1e-8):

        inputs_shape = input_seq.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(input_seq, [-1], keepdims=True)
        beta = tf.zeros(params_shape)
        gamma = tf.ones(params_shape)
        normalized = (input_seq - mean) / ((variance + epsilon) ** .5)
        output = gamma * normalized + beta

        return output


    def multihead_attention(self, queries, keys):

        # Linear projections
        Q = tf.keras.layers.Dense(self.attention_dim, activation=None)(queries) # (N, T_q, C)
        K = tf.keras.layers.Dense(self.attention_dim, activation=None)(keys) # (N, T_k, C)
        V = tf.keras.layers.Dense(self.attention_dim, activation=None)(keys) # (N, T_k, C)

        # --- MULTI HEAD ---
        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.attention_num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, self.attention_num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, self.attention_num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)


        # --- SCALED DOT PRODUCT ---
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [self.attention_num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Future blinding (Causality)
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [self.attention_num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.keras.layers.Dropout(self.dropout_rate)(outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # --- MULTI HEAD ---
        # concat heads
        outputs = tf.concat(tf.split(outputs, self.attention_num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        return outputs


    def point_wise_feed_forward(self, input_seq):

        output = tf.keras.layers.Conv1D(filters=self.conv_dims[0], kernel_size=1, activation='relu', use_bias=True)(input_seq)
        output = tf.keras.layers.Dropout(self.dropout_rate)(output)

        output = tf.keras.layers.Conv1D(filters=self.conv_dims[1], kernel_size=1, activation=None, use_bias=True)(output)
        output = tf.keras.layers.Dropout(self.dropout_rate)(output)

        # Residual connection
        output += input_seq

        return output


