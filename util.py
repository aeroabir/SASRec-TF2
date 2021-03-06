import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
import sys
import pickle

from tqdm import tqdm

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

from metric import ndcg_at_k, recall_at_k

COL_USER = "UserId"
COL_ITEM = "ItemId"
COL_RATING = "Rating"
COL_PREDICTION = "Rating"


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
                idx = word_index.index(word) + 1
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                    :embedding_dim
                ]
    count_missing = len(set(word_index) - all_words)
    if count_missing > 0:
        print(f"!!! {count_missing} words could not be mapped")
    return embedding_matrix, all_words


def text_processing(args):
    data_dir = "data/"
    filename = args.dataset + "_item_description.txt"
    glove_dir = "/recsys_data/datasets/glove"
    glove_file = "glove.6B.50d.txt"
    maxlen = args.text_maxlen
    vocab_size = args.vocab_size
    embedding_dim = args.text_embed

    print(f"Processing for textual features")
    with open(os.path.join(data_dir, filename), "r") as fr:
        docs = fr.readlines()
    tokenizer = Tokenizer(num_words=vocab_size - 1, lower=True, split=" ")  # 1 ... 4999
    # tokenizer = Tokenizer(num_words=vocab_size, lower=True, split=' ', oov_token='<OOV>')
    tokenizer.fit_on_texts(docs)
    print(f"Number of words found: {len(tokenizer.word_index)}")
    vocab = [k for k, v in tokenizer.word_index.items() if v < vocab_size]  # 1 ... 4999
    tensor = tokenizer.texts_to_sequences(docs)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding="post", maxlen=maxlen
    )
    print(f"Tokenized each item description", tensor.shape)

    # add a zero row
    num_items, seq_len = tensor.shape
    big_tensor = np.zeros((num_items + 1, seq_len))
    big_tensor[1 : num_items + 1, :] = tensor

    embedding_matrix, glove_vocab = create_embedding_matrix(
        os.path.join(glove_dir, glove_file), vocab, embedding_dim, vocab_size
    )

    item_embeddings = np.zeros((num_items + 1, embedding_matrix.shape[1]))
    for item in tqdm(range(1, num_items + 1)):
        word_indices = big_tensor[item, :]
        word_indices = [int(i) for i in word_indices if i != 0]
        if len(word_indices) > 0:
            word_vectors = embedding_matrix[word_indices, :]
            mean_vector = word_vectors.mean(axis=0)
            item_embeddings[item, :] = mean_vector
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
    colsep = "\t"

    sample = pd.read_csv(f"data/{fname}.txt", sep=colsep, nrows=10)
    ncol = sample.shape[1]
    if ncol == 1:
        raise ValueError("Not enough data to unpack!!")

    # assume user/item index starting from 1
    f = open("data/%s.txt" % fname, "r")
    for line in f:
        if ncol == 2:
            u, i = line.rstrip().split(colsep)
        elif ncol == 3:
            u, i, timestamp = line.rstrip().split(colsep)
        elif ncol == 4:
            u, i, _, _ = line.rstrip().split(colsep)
        else:
            raise ValueError("Unknown number of columns")

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


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(
            map(lambda x: [item_map[x[0]], time_map[x[1]]], items)
        )

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(
            map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items)
        )
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)


def data_partition_with_time(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    print("Preparing data...")
    f = open("data/%s.txt" % fname, "r")
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split("\t")
        except:
            u, i, timestamp = line.rstrip().split("\t")
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open("data/%s.txt" % fname, "r")
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split("\t")
        except:
            u, i, timestamp = line.rstrip().split("\t")
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

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
    print("Preparing done...")
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]


def predict(model, dataset, extras):
    """Predict the next best item
    based on the entire interaction history

    Taking a huge time if there are too many
    users and products since this is done one
    at a time.

    """

    args = extras[0]
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    next_items = {}
    for u in tqdm(users, ncols=70, leave=False, unit="b"):

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = test[u][0]  # last item
        idx -= 1
        seq[idx] = valid[u][0]  # last but one item (coming from validation)
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # test over all the items
        item_idx = [ii for ii in range(1, itemnum + 1)]
        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])

        # we need to break the items into a batch of 101
        b = args.num_neg_test + 1
        num_examples = len(item_idx)
        num_steps = int(num_examples / b)
        rem = int(num_examples % b)
        if rem > 0:
            num_steps += 1
        start, end = 0, b
        all_preds = []
        for ii in range(num_steps):
            items_ii = item_idx[start:end]
            # print(ii, start, end, len(items_ii))
            if len(items_ii) < b:
                items_ii += [items_ii[-1]] * (b - len(items_ii))
                # print(ii, len(items_ii), "appended")
            inputs["candidate"] = np.array([items_ii])
            pred_ii = model.predict(inputs)  # (1, b)
            all_preds.append(pred_ii)

            start += b
            end += b
            if end > num_examples:
                end = num_examples
        all_preds = tf.concat(all_preds, -1)
        all_preds = all_preds[0, 0:itemnum]
        all_preds = -1.0 * all_preds
        all_preds = np.array(all_preds)

        next_item = all_preds.argmax()
        next_items[u] = next_item

    return next_items


def evaluate(model, dataset, extras):

    args = extras[0]
    if args.add_embeddings == 1:
        item_df = extras[1]
        # item-number to category-id
        item2cat = [ii + 1 for ii in item_df["cid"].tolist()]
        item2cat.insert(0, 0)

        def map_category(source, mappings):
            target = []
            for s in source:
                smap = [mappings[i] for i in s]
                target.append(smap)
            return target

    if args.add_time == 1:
        res = evaluate_with_time(model, dataset, args)
        return res

    elif args.add_history == 1:
        res = evaluate_with_graph(model, dataset, args)
        return res

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    df_true = {
        COL_USER: [],
        COL_ITEM: [],
        COL_RATING: [],
    }

    df_pred = {
        COL_USER: [],
        COL_ITEM: [],
        COL_PREDICTION: [],
    }

    for u in tqdm(users, ncols=70, leave=False, unit="b"):

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(args.num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])
        inputs["candidate"] = np.array([item_idx])

        if args.add_embeddings == 1:
            seq_cid = map_category([seq], item2cat)
            item_idx_cid = map_category([item_idx], item2cat)
            inputs["input_seq_c"] = np.array(seq_cid)
            inputs["candidates_c"] = np.array(item_idx_cid)

        # inverse to get descending sort
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        # user_list = [u for _ in range(args.num_neg_test + 1)]
        # item_list = copy.deepcopy(item_idx)
        # rating_list = [1] + [0] * args.num_neg_test
        # pred_list = predictions.tolist()

        # df_true[COL_USER] += user_list
        # df_true[COL_ITEM] += item_list
        # df_true[COL_RATING] += rating_list

        # df_pred[COL_USER] += user_list
        # df_pred[COL_ITEM] += item_list
        # df_pred[COL_PREDICTION] += pred_list

        # double sorting trick to get the rank
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.', end="")
        #     sys.stdout.flush()

        # df_true = pd.DataFrame(df_true)
        # df_pred = pd.DataFrame(df_pred)
        # ndcg = ndcg_at_k(
        #     rating_true=df_true,
        #     rating_pred=df_pred,
        #     col_user=COL_USER,
        #     col_item=COL_ITEM,
        #     col_rating=COL_RATING,
        #     col_prediction=COL_PREDICTION,
        #     relevancy_method="top_k",
        #     k=10,
        #     threshold=101,
        # )
        # recall = recall_at_k(
        #     rating_true=df_true,
        #     rating_pred=df_pred,
        #     col_user=COL_USER,
        #     col_item=COL_ITEM,
        #     col_rating=COL_RATING,
        #     col_prediction=COL_PREDICTION,
        #     relevancy_method="top_k",
        #     k=10,
        #     threshold=101,
        # )
        # print(f"\nNDCG@10: {ndcg}, Hit@10: {recall}")
        # print(f"NDCG@10: {NDCG}, Hit@10: {HT}")
        # with open("data/sample.pkl", "wb") as handle:
        #     pickle.dump((df_true, df_pred), handle, protocol=pickle.HIGHEST_PROTOCOL)
        # sys.exit("TEST")

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, extras):

    args = extras[0]
    if args.add_embeddings == 1:
        item_df = extras[1]
        # item-number to category-id
        item2cat = [ii + 1 for ii in item_df["cid"].tolist()]
        item2cat.insert(0, 0)

        def map_category(source, mappings):
            target = []
            for s in source:
                smap = [mappings[i] for i in s]
                target.append(smap)
            return target

    if args.add_time == 1:
        res = evaluate_valid_with_time(model, dataset, args)
        return res

    if args.add_history == 1:
        res = evaluate_valid_with_graph(model, dataset, args)
        return res

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit="b"):
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(args.num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])
        inputs["candidate"] = np.array([item_idx])
        # if args.text_features == 1:
        #     inputs['inp_seq_tokens'] = item2text[inputs['input_seq'], :]
        #     inputs['candidate_tokens'] = item2text[inputs['candidate'], :]

        if args.add_embeddings == 1:
            seq_cid = map_category([seq], item2cat)
            item_idx_cid = map_category([item_idx], item2cat)
            inputs["input_seq_c"] = np.array(seq_cid)
            inputs["candidate_c"] = np.array(item_idx_cid)

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


def computeRePos(time_seq, time_span):

    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc="Preparing relation matrix"):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train


def evaluate_with_time(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit="b"):

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(args.num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)

        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])
        inputs["candidate"] = np.array([item_idx])
        inputs["time_matrix"] = np.array([time_matrix])

        # inverse to get descending sort
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_valid_with_time(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit="b"):
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)
        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])
        inputs["candidate"] = np.array([item_idx])
        inputs["time_matrix"] = np.array([time_matrix])

        # inverse to get descending sort
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def fill_zeros(lst):
    lst2 = []
    for elem in lst:
        if elem == "0":
            lst2.append([0])
        else:
            lst2.append([int(h) for h in elem.split(",")])
    return lst2


def data_partition_with_graph(fname):
    """
    Read data with past user & item history associated with each item
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open("data/%s.txt" % fname, "r")
    for line in f:
        elements = line.rstrip().split("\t")
        u = elements[0]
        i = elements[1]
        u = int(u)
        i = int(i)
        # only item history
        # hist = elements[3]
        # if hist == "0":
        #     hist = [0]
        # else:
        #     hist = [int(h) for h in hist.split(",")]

        # both user and item history
        hist = [elements[2], elements[3]]
        hist = fill_zeros(hist)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append((i, hist))

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


def evaluate_with_graph(model, dataset, args):

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit="b"):

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        his = np.zeros([args.maxlen, args.user_len], dtype=np.int32)
        his_u = np.zeros([args.maxlen, args.user_len], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]

        if all(isinstance(el, list) for el in valid[u][0][1]):
            hist_u, hist_i = valid[u][0][1]
            num_users = min(len(hist_i), args.user_len)
            his[idx][args.user_len - num_users :] = hist_i[:num_users]

            num_users = min(len(hist_u), args.user_len)
            his_u[idx][args.user_len - num_users :] = hist_u[:num_users]
        else:
            num_users = min(len(valid[u][0][1]), args.user_len)
            his[idx][args.user_len - num_users :] = valid[u][0][1][:num_users]

        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            if all(isinstance(el, list) for el in i[1]):
                hist_u, hist_i = i[1]
                num_users = min(len(hist_i), args.user_len)
                his[idx][args.user_len - num_users :] = hist_i[:num_users]

                num_users = min(len(hist_u), args.user_len)
                his_u[idx][args.user_len - num_users :] = hist_u[:num_users]
            else:
                num_users = min(len(i[1]), args.user_len)
                his[idx][args.user_len - num_users :] = i[1][:num_users]

            idx -= 1
            if idx == -1:
                break
        rated = set([x[0] for x in train[u]])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(args.num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])
        inputs["candidate"] = np.array([item_idx])
        inputs["user_history"] = np.array([his_u])
        inputs["item_history"] = np.array([his])

        # inverse to get descending sort
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        # double sorting trick to get the rank
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_valid_with_graph(model, dataset, args):

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit="b"):
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        his = np.zeros([args.maxlen, args.user_len], dtype=np.int32)
        his_u = np.zeros([args.maxlen, args.user_len], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            if all(isinstance(el, list) for el in i[1]):
                hist_u, hist_i = i[1]
                num_elems = min(len(hist_i), args.user_len)
                his[idx][args.user_len - num_elems :] = hist_i[:num_elems]

                num_users = min(len(hist_u), args.user_len)
                his_u[idx][args.user_len - num_users :] = hist_u[:num_users]

            else:
                num_users = min(len(i[1]), args.user_len)
                his[idx][args.user_len - num_users :] = i[1][:num_users]
            idx -= 1
            if idx == -1:
                break

        rated = set([x[0] for x in train[u]])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(args.num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])
        inputs["candidate"] = np.array([item_idx])
        inputs["user_history"] = np.array([his_u])
        inputs["item_history"] = np.array([his])

        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user
