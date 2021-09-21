import os
import sys
import copy
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_embedding_matrix(filepath, word_index, embedding_dim, vocab_size):
    # vocab_size = len(word_index) + 1  
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    all_words = set()
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            all_words.add(word)
            if word in word_index:
                idx = word_index.index(word)+1 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    count_missing = len(set(word_index) - all_words)
    if count_missing > 0:
        print(f"!!! {count_missing} words could not be mapped")
    return embedding_matrix, all_words


def text_processing(args):
    data_dir = 'data/'
    filename = args.dataset + "_item_description.txt"
    glove_dir = "/recsys_data/datasets/glove"
    glove_file = 'glove.6B.50d.txt'
    maxlen = args.text_maxlen
    vocab_size = args.vocab_size
    embedding_dim = args.text_embed

    print(f"Processing for textual features")
    with open(os.path.join(data_dir, filename), 'r') as fr:
        docs = fr.readlines()
    tokenizer = Tokenizer(num_words=vocab_size-1, lower=True, split=' ')  # 1 ... 4999
    # tokenizer = Tokenizer(num_words=vocab_size, lower=True, split=' ', oov_token='<OOV>')
    tokenizer.fit_on_texts(docs)
    print(f"Number of words found: {len(tokenizer.word_index)}")
    vocab = [k for k,v in tokenizer.word_index.items() if v < vocab_size]  # 1 ... 4999
    tensor = tokenizer.texts_to_sequences(docs)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=maxlen)
    print(f"Tokenized each item description", tensor.shape)

    # add a zero row
    num_items, seq_len = tensor.shape
    big_tensor = np.zeros((num_items+1, seq_len))
    big_tensor[1:num_items+1, :] = tensor

    embedding_matrix, glove_vocab = create_embedding_matrix(os.path.join(glove_dir, glove_file),
                                                            vocab,  
                                                            embedding_dim,
                                                            vocab_size)

    item_embeddings = np.zeros((num_items+1, embedding_matrix.shape[1]))
    for item in tqdm(range(1, num_items+1)):
        word_indices = big_tensor[item, :]
        word_indices = [int(i) for i in word_indices if i != 0]
        if len(word_indices) > 0:
            word_vectors = embedding_matrix[word_indices, :]
            mean_vector = word_vectors.mean(axis=0)
            item_embeddings[item,:] = mean_vector
        else:
            print(f"Missing embedding for item-{item}")

    print(f"Text based item embedding matrix", item_embeddings.shape)
    return item_embeddings  # big_tensor, embedding_matrix


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def data_partition_with_time(fname, sep=" "):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    Items = set()
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    
    for line in f:
        u, i, t = line.rstrip().split(sep)
        User[u].append((i, t))
        Items.add(i)

    for user in User.keys():
        # sort by time
        items = sorted(User[user], key=lambda x: x[1])
        # keep only the items
        items = [x[0] for x in items]
        User[user] = items
        nfeedback = len(User[user])
        if nfeedback == 1:
            del User[user]
            continue
        elif nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    
    usernum = len(User)
    itemnum = len(Items)
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in tqdm(users, ncols=70, leave=False, unit='b'):

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(args.num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs['user'] = np.expand_dims(np.array([u]), axis=-1)
        inputs['input_seq'] = np.array([seq])
        inputs['candidate'] = np.array([item_idx])
        # if args.text_features == 1:
        #     inputs['inp_seq_tokens'] = item2text[inputs['input_seq'], :]
        #     inputs['candidate_tokens'] = item2text[inputs['candidate'], :]

        # inverse to get descending sort
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.', end="")
        #     sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit='b'):
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(args.num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs['user'] = np.expand_dims(np.array([u]), axis=-1)
        inputs['input_seq'] = np.array([seq])
        inputs['candidate'] = np.array([item_idx])
        # if args.text_features == 1:
        #     inputs['inp_seq_tokens'] = item2text[inputs['input_seq'], :]
        #     inputs['candidate_tokens'] = item2text[inputs['candidate'], :]

        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.', end="")
        #     sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
