import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from tqdm import tqdm

# from model import Model
# from model2 import SASREC
from model3 import SASREC
from model_sse_pt import SSEPT
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def create_combined_dataset(u, seq, pos, neg, seq_max_len):
    inputs = {}
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq,
            padding='pre',
            truncating='pre',
            maxlen=seq_max_len)
    pos = tf.keras.preprocessing.sequence.pad_sequences(pos,
            padding='pre',
            truncating='pre',
            maxlen=seq_max_len)
    neg = tf.keras.preprocessing.sequence.pad_sequences(neg,
            padding='pre',
            truncating='pre',
            maxlen=seq_max_len)

    # print(seq.shape, pos.shape, neg.shape)
    # inputs['input_seq'] = np.concatenate([seq, seq], axis=0)
    # inputs['candidate'] = np.concatenate([pos, neg], axis=0)
    inputs['users'] = np.expand_dims(np.array(u), axis=-1)
    inputs['input_seq'] = seq
    inputs['positive'] = pos
    inputs['negative'] = neg

    target = np.concatenate([np.repeat(1, seq.shape[0] * seq.shape[1]), 
                             np.repeat(0, seq.shape[0] * seq.shape[1])], axis=0)
    target = np.expand_dims(target, axis=-1)
    return inputs, target


def build_tf_dataset(dataset_dict: dict, seq_max_len: int) -> [dict, np.array]:

    inputs = {}
    inputs['input_seq'] = tf.keras.preprocessing.sequence.pad_sequences(
        dataset_dict['history'], padding='post', truncating='post', maxlen=seq_max_len)
    inputs['candidate'] = np.array(dataset_dict['candidate'])

    # print(inputs)
    # sys.exit("HH")

    output = np.array(dataset_dict['label'])

    return inputs, output


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--num_neg_test', default=100, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

# if args.dataset in ['ae']:
#     dataset = data_partition_with_time(args.dataset, "\t")
# else:
dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_steps = int(len(user_train) / args.batch_size)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('%g Users and %g items' % (usernum, itemnum))
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

# model = SASREC(item_num=itemnum,
#                seq_max_len=args.maxlen,
#                num_blocks=args.num_blocks,
#                embedding_dim=args.hidden_units,
#                attention_dim=args.hidden_units,
#                attention_num_heads=args.num_heads,
#                dropout_rate=args.dropout_rate,
#             #    conv_dims = kwargs.get("conv_dims", [100, 100])
#                l2_reg=args.l2_emb,
#                num_neg_test=args.num_neg_test
# )

model = SSEPT(item_num=itemnum,
              user_num=usernum,
              seq_max_len=args.maxlen,
              num_blocks=args.num_blocks,
              user_embedding_dim=args.hidden_units,
              item_embedding_dim=args.hidden_units,
              attention_dim=args.hidden_units,
              attention_num_heads=args.num_heads,
              dropout_rate=args.dropout_rate,
              l2_reg=args.l2_emb,
              num_neg_test=args.num_neg_test
)

# optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, decay=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-7)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

def loss_function(pos_logits, neg_logits, istarget):
    pos_logits = pos_logits[:,0]
    neg_logits = neg_logits[:,0]

    # for logits
    loss = tf.reduce_sum(
        - tf.math.log(tf.math.sigmoid(pos_logits) + 1e-24) * istarget -
        tf.math.log(1 - tf.math.sigmoid(neg_logits) + 1e-24) * istarget
    ) / tf.reduce_sum(istarget)

    # for probabilities
    # loss = tf.reduce_sum(
    #         - tf.math.log(pos_logits + 1e-24) * istarget -
    #         tf.math.log(1 - neg_logits + 1e-24) * istarget
    # ) / tf.reduce_sum(istarget)
    reg_loss = tf.compat.v1.losses.get_regularization_loss()
    # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    # loss += sum(reg_losses)
    loss += reg_loss
    return loss

def loss_function_(real, pred):
    # real and pred are (batch * seq_len, 1)
    # mask = tf.math.equal(pred, 0)
    mask = tf.math.logical_not(tf.math.equal(pred, 0))
    real = real[mask]
    pred = pred[mask]
    loss_ = loss_object(real, pred)
    # mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask
#     return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    return tf.reduce_mean(loss_)

def accuracy_function(real, pred):
    pred_class = tf.where(pred > 0.5, 1, 0)
    accuracies = tf.equal(real, pred_class)
    # accuracies = tf.equal(tf.argmax(real, axis=1), tf.argmax(pred, axis=1))
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    return tf.reduce_mean(accuracies)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

train_step_signature = [
    {'users': tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
     'input_seq': tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
     'positive': tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
     'negative': tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64)},
    tf.TensorSpec(shape=(None, 1), dtype=tf.int64)
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        pos_logits, neg_logits, loss_mask = model(inp, training=True)
        loss = loss_function(pos_logits, neg_logits, loss_mask)
        # loss = loss_function_(tar, predictions)
        # loss = model.loss_function(*predictions)

    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    # train_accuracy(accuracy_function(tar, predictions))
    return loss


sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

T = 0.0
t0 = time.time()

# try:
for epoch in range(1, args.num_epochs + 1):

    step_loss = []
    train_loss.reset_states()
    for step in tqdm(range(num_steps), total=num_steps, ncols=70, leave=False, unit='b'):

        u, seq, pos, neg = sampler.next_batch()

        inputs, target = create_combined_dataset(u, seq, pos, neg, args.maxlen)
        # out = model(inputs, training=True)
        # print(out)
        # sys.exit("KK")

        loss = train_step(inputs, target)
        step_loss.append(loss)
        # with tf.GradientTape() as tape:
        #     pos_logits, neg_logits, istarget = model(inputs, training=True)
        #     loss = loss_function(pos_logits, neg_logits, istarget)

        # grads = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"Epoch: {epoch}, Loss: {np.mean(step_loss):.3f}, {train_loss.result():.3f}")
        
    if epoch % 20 == 0:
        t1 = time.time() - t0
        T += t1
        print('Evaluating...')
        t_test = evaluate(model, dataset, args)
        t_valid = evaluate_valid(model, dataset, args)
        print(f"\nepoch: {epoch}, time: {T}, valid (NDCG@10: {t_valid[0]}, HR@10: {t_valid[1]})")
        print(f"epoch: {epoch}, time: {T},  test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})")

        f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        f.flush()
        t0 = time.time()

t_test = evaluate(model, dataset, args)
print(f"\nepoch: {epoch}, test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})")
# except:
#     sampler.close()
#     f.close()
#     exit(1)

f.close()
sampler.close()
print("Done")

"""
    Results:
    python main-tf2.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.1 --lr=0.001 --hidden_units=100
    epoch: 200, time: 1023.7011332511902, valid (NDCG@10: 0.6056608512323377, HR@10: 0.8316225165562914)
    epoch: 200, time: 1023.7011332511902,  test (NDCG@10: 0.578710624727615, HR@10: 0.8079470198675497)

    python main-tf2.py --dataset=ae --train_dir=default --maxlen=50 --dropout_rate=0.2 --lr=0.001 --hidden_units=100
    63114 Users and 85930 items
    average sequence length: 13.04

    python main-tf2.py --dataset=ae --train_dir=default --maxlen=50 --dropout_rate=0.5 --lr=0.001 --hidden_units=100 --num_epochs=50
    epoch: 40, time: 1199.8057270050049, valid (NDCG@10: 0.38133518193540433, HR@10: 0.5888)
    epoch: 40, time: 1199.8057270050049,  test (NDCG@10: 0.3333367519011616, HR@10: 0.5251)
    epoch: 50,                            test (NDCG@10: 0.3410992805652115, HR@10: 0.5336)

    python main-tf2.py --dataset=Beauty --train_dir=default --maxlen=50 --dropout_rate=0.1 --lr=0.001 --hidden_units=100
    epoch: 40, time: 729.5262145996094, valid (NDCG@10: 0.28800812778906604, HR@10: 0.42751006088123)
    epoch: 40, time: 729.5262145996094,  test (NDCG@10: 0.2554444303109231, HR@10: 0.38506998444790047)
    nan issue
"""
"""
        # # train with positive samples
        # dataset_dict = {'history': seq, 'candidate': pos, 'label': np.repeat(1, args.batch_size)}
        # inputs, output = build_tf_dataset(dataset_dict, args.maxlen)

        # # for k in inputs:
        # #     print(k, inputs[k].shape)
        # # print(model(inputs, training=False).shape)
        # # sys.exit("HERE")
        
        # with tf.GradientTape() as tape:
        #     logits = model.model(inputs, training=True)
        #     loss_value = loss_object(output, logits)
        
        # grads = tape.gradient(loss_value, model.trainable_weights)
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # # train with negative samples
        # dataset_dict = {'history': seq, 'candidate': neg, 'label': np.repeat(0, args.batch_size)}
        # inputs, output = build_tf_dataset(dataset_dict, args.maxlen)

        # with tf.GradientTape() as tape:
        #     logits = model.model(inputs, training=True)
        #     loss_value = loss_object(output, logits)
"""