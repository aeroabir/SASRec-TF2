import random

import category_encoders as ce
import pandas as pd
import numpy as np


def token_col_encode(df: pd.DataFrame, cols: list) -> [pd.DataFrame, dict]:

    ordinal_encoders = {}
    for col in cols:
        # ''をカテゴリーと認識させないため、一旦NaNに置き換える
        df[col] = df[col].replace('', np.nan)

        # ordinal encoding
        ordinal_encoders[col] = ce.OrdinalEncoder(cols=[col], handle_missing='return_nan')
        df = ordinal_encoders[col].fit_transform(df)

        # NaNはpaddingと同じ0として扱う
        df[col] = df[col].fillna(0).astype(int)
        # 未知語-1は最大id+1を付与
        vocab_count = len(ordinal_encoders[col].get_params()['mapping'][0]['mapping'])
        df[col] = df[col].replace(-1, vocab_count)

    return [df, ordinal_encoders]


def aggregate_features(df: pd.DataFrame, user_col: str, sort_cols: list, behavior_key_col: str,
                       behavior_category_cols: list,
                       seq_max_len: int, user_category_cols: list = None, context_cols: list = None) -> dict:
    """
    FIXME 連続値対応していない(negative生成を要素数を元に行うため)
    エンコード後のログデータから行動系列含む特徴量として整形
    正例と1:1になるように負例をランダムに生成
    :param df:
    :param user_col: group化するときのキーとなるカラム（通常はuser_id）
    :param sort_cols: sortするためのカラム
    :param behavior_key_col: 基準となるbehaviorのカラム
    :param behavior_category_cols: behavior_key_col以外の行動系列として使用するカラム
    :param seq_max_len: 行動系列の系列長
    :param user_category_cols: user_col以外のuser属性特徴量（userに結びつく）
    :param context_cols: context特徴量（labelのアイテムに結びつく）
    :return:
    """

    # user_col以外のカラムをまとめる
    # hist_cols: user_col以外
    # category_cols: user_col, 行動系列以外
    hist_cols = [behavior_key_col]
    behavior_cols = [behavior_key_col]
    category_cols = []
    if behavior_category_cols:
        behavior_cols.extend(behavior_category_cols)
        category_cols.extend(behavior_category_cols)
    if user_category_cols:
        category_cols.extend(user_category_cols)
    if context_cols:
        category_cols.extend(context_cols)
    if category_cols:
        hist_cols.extend(category_cols)

    # 特徴量格納用dictionary用意
    dataset_dict = {}
    dataset_dict['label'] = []
    dataset_dict['category_' + user_col] = []
    for behavior_col in behavior_cols:
        dataset_dict['candidate_' + behavior_col] = []
        dataset_dict['behavior_' + behavior_col] = []
    if category_cols:
        for category_col in category_cols:
            dataset_dict['category_' + category_col] = []

    # sort_cols で昇順に並び替える
    df = df.sort_values(by=sort_cols, ascending=True)
    df = df.drop(sort_cols, axis=1)

    # negative生成のため各カラムの要素数を取得
    item_count = {}
    for hist_col in hist_cols:
        item_count[hist_col] = df[hist_col].max()

    # negative生成用マスタ作成
    df_master = df[behavior_cols].drop_duplicates()

    for user, hist in df.groupby(user_col):

        negative_list_dict = {}
        positive_list_dict = {}

        for hist_col in hist_cols:

            # positiveのリストを作成
            # seq_max_len+1を超える場合は最初の方のレコードを切り捨て
            positive_list = hist[hist_col].tolist()
            positive_list_dict[hist_col] = positive_list[-(seq_max_len + 1):]

            # negativeのリストを作成
            negative_list_dict[hist_col] = \
                [_generate_negative(positive_list, item_count[hist_col]) for _ in range(len(positive_list))]

        # 行動ごとにレコードを作成
        # 行動データが項目間で同じ長さであることが前提となる
        for i in range(1, len(positive_list_dict[behavior_cols[0]])):
            # positiveレコードの生成
            dataset_dict['label'].append(1)
            dataset_dict['category_' + user_col].append(user)

            for behavior_col in behavior_cols:
                dataset_dict['candidate_' + behavior_col].append(positive_list_dict[behavior_col][i])
                dataset_dict['behavior_' + behavior_col].append(positive_list_dict[behavior_col][:i])

            for category_col in category_cols:
                dataset_dict['category_' + category_col].append(positive_list_dict[category_col][i])

            # negativeレコードの生成
            dataset_dict['label'].append(0)
            dataset_dict['category_' + user_col].append(user)

            dataset_dict['candidate_' + behavior_key_col].append(negative_list_dict[behavior_key_col][i])
            dataset_dict['behavior_' + behavior_key_col].append(positive_list_dict[behavior_key_col][:i])

            for behavior_category_col in behavior_category_cols:

                # 対象のnegativeに対応する行動系列を取得
                value = df_master[df[behavior_key_col] ==
                                  negative_list_dict[behavior_key_col][i]][behavior_category_col].values()[0]
                dataset_dict['candidate_' + behavior_category_col].append(value)

                dataset_dict['behavior_' + behavior_category_col].append(positive_list_dict[behavior_category_col][:i])

            for category_col in category_cols:
                dataset_dict['category_' + category_col].append(negative_list_dict[category_col][i])

    return dataset_dict


def _generate_negative(positive_list, item_count):
    negative = positive_list[0]

    while negative in positive_list:
        negative = random.randint(0, item_count-1)

    return negative