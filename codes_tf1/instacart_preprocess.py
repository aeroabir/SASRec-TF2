"""
insta_cartデータ前処理
"""

import os
import tarfile

import pandas as pd


def initialize_data(tarfile_path, directory, filename, use_columns):

    if not os.path.isfile(os.path.join(directory, filename+'.csv')):

        unpack_tar_gz(tarfile_path, directory)
        df = make_data(directory, use_columns)

        df.to_csv(os.path.join(directory, filename + '.tsv'), sep='\t')


def unpack_tar_gz(tarfile_path, directory):

    with tarfile.open(tarfile_path, 'r:gz') as tf:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf, path=directory)


def make_data(directory, use_columns):

    data_dir = os.path.join(directory, 'instacart_2017_05_01')

    df_orders = pd.read_csv(os.path.join(data_dir, 'orders.csv'))
    df_products = pd.read_csv(os.path.join(data_dir, 'products.csv'))
    df_aisles = pd.read_csv(os.path.join(data_dir, 'aisles.csv'))
    df_departments = pd.read_csv(os.path.join(data_dir, 'departments.csv'))

    df_order_products_prior = pd.read_csv(os.path.join(data_dir, 'order_products__prior.csv'))
    df_orders_prior = df_orders[df_orders.eval_set == 'prior']
    df_orders_prior = pd.merge(df_orders_prior, df_order_products_prior, on='order_id', how='inner')

    df_order_products_train = pd.read_csv(os.path.join(data_dir, 'order_products__train.csv'))
    df_orders_train = df_orders[df_orders.eval_set == 'train']
    df_orders_train = pd.merge(df_orders_train, df_order_products_train, on='order_id', how='inner')

    df = pd.concat([df_orders_prior, df_orders_train])

    df = pd.merge(df, df_products, on='product_id', how='inner')
    df = pd.merge(df, df_aisles, on='aisle_id', how='inner')
    df = pd.merge(df, df_departments, on='department_id', how='inner')

    df = df[use_columns]

    df = df.fillna('').astype(str)

    return df


def main():

    tarfile_path = 'data/packed/instacart_online_grocery_shopping_2017_05_01.tar.gz'
    data_dir = 'data/unpacked'
    data_filename = 'data'
    use_columns = ['user_id', 'order_number', 'add_to_cart_order', 'order_dow', 'order_hour_of_day',
                   'days_since_prior_order', 'product_name', 'aisle', 'department']

    initialize_data(tarfile_path, data_dir, data_filename, use_columns)


if __name__ == '__main__':
    main()