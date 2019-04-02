import os
import pickle
import argparse

import numpy as np
import pandas as pd

def load_data(fpath, sep='::', engine='python'):
    # Load data
    df = pd.read_csv(fpath, sep=sep, engine=engine,
            names=['user', 'item', 'rate', 'time'])
    df = df[df['user']>=3]
    return df

def convert_unique_idx(df, column_name):
    column_dict = {x:i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df

def create_user_list(df, user_size):
    user_list = [dict() for u in range(user_size)]
    for idx, row in df.iterrows():
        user_list[row['user']][row['item']] = row['time']
    return user_list

def split_train_test(user_list, test_size=0.2, time_order=False):
    train_user_list = [None] * len(user_list)
    test_user_list  = [None] * len(user_list)
    for user, item_dict in enumerate(user_list):
        if time_order:
            # Choose latest item
            item = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
            latest_item = item[:len(item)//5]
            assert max(item_dict.values()) == latest_item[0][1]
            test_item = set(map(lambda x: x[0], latest_item))
        else:
            # Random select
            test_item = set(np.random.choice(list(item_dict.keys()),
                                             size=len(item_dict)//5,
                                             replace=False))
        assert len(test_item) > 0, "No test item for user %d" % user
        test_user_list[user] = test_item
        train_user_list[user] = set(item_dict.keys()) - test_item
    return train_user_list, test_user_list

def create_pair(user_list):
    pair = []
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item) for item in item_set])
    return pair

if __name__=='__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str,
            default=os.path.join('data', 'ml-1m', 'ratings.dat'),
            help="File path for raw data")
    parser.add_argument('--output_data', type=str,
            default=os.path.join('preprocessed', 'bpr-movielens-1m.pickle'),
            help="File path for preprocessed data")
    parser.add_argument('--test_size', type=float, default=0.2,
            help="Proportion for training and testing split")
    parser.add_argument('--time_order', action='store_true',
            help="Proportion for training and testing split")
    args = parser.parse_args()

    df = load_data(args.input_data)
    df = convert_unique_idx(df, 'user')
    df = convert_unique_idx(df, 'item')
    user_size = len(df['user'].unique())
    item_size = len(df['item'].unique())

    total_user_list = create_user_list(df, user_size)
    train_user_list, test_user_list = split_train_test(total_user_list,
                                                       test_size=args.test_size,
                                                       time_order=args.time_order)

    train_w = np.zeros((user_size, item_size))
    for u, itemset in enumerate(train_user_list):
        for i in itemset:
            train_w[u, i] = 1
    test_w = np.zeros((user_size, item_size))
    for u, itemset in enumerate(test_user_list):
        for i in itemset:
            test_w[u, i] = 1
    train_pair = create_pair(train_user_list)

    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_w': train_w, 'test_w': test_w, 'train_pair': train_pair}
    dirname = os.path.dirname(os.path.abspath(args.output_data))
    os.makedirs(dirname, exist_ok=True)
    with open(args.output_data, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
