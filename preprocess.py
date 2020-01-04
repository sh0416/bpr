import os
import random
import pickle
import argparse

import numpy as np
import pandas as pd


class DatasetLoader(object):
    def load(self):
        raise NotImplementedError


class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.dat')

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'rate', 'time'])
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        return df


class MovieLens20M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.csv')

    def load(self):
        df = pd.read_csv(self.fpath,
                         sep=',',
                         names=['user', 'item', 'rate', 'time'],
                         usecols=['user', 'item', 'time'],
                         skiprows=1)
        return df


class Gowalla(DatasetLoader):
    """Work In Progress"""
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'loc-gowalla_totalCheckins.txt')

    def load(self):
        df = pd.read_csv(self.fpath,
                         sep='\t',
                         names=['user', 'time', 'latitude', 'longitude', 'item'],
                         usecols=['user', 'item', 'time'])
        df_size, df_nxt_size = 0, len(df)
        while df_size != df_nxt_size:
            # Update
            df_size = df_nxt_size

            # Remove user which doesn't contain at least five items to guarantee the existance of `test_item`
            groupby_user = df.groupby('user')['item'].nunique()
            valid_user = groupby_user.index[groupby_user >= 15].tolist()
            df = df[df['user'].isin(valid_user)]
            df = df.reset_index(drop=True)

            # Remove item which doesn't contain at least five users
            groupby_item = df.groupby('item')['user'].nunique()
            valid_item = groupby_item.index[groupby_item >= 15].tolist()
            df = df[df['item'].isin(valid_item)]
            df = df.reset_index(drop=True)

            # Update
            df_nxt_size = len(df)

        print('User distribution')
        print(df.groupby('user')['item'].nunique().describe())
        print('Item distribution')
        print(df.groupby('item')['user'].nunique().describe())
        return df


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df


def create_user_list(df, user_size):
    user_list = [list() for u in range(user_size)]
    for row in df.itertuples():
        user_list[row.user].append((row.time, row.item))
    return user_list


def split_train_test(user_list, test_size=0.2, time_order=False):
    train_user_list = [None] * len(user_list)
    test_user_list = [None] * len(user_list)
    for user, item_list in enumerate(user_list):
        if time_order:
            # Choose latest item
            item_list = sorted(item_list, key=lambda x: x[0], reverse=True)
        else:
            # Random shuffle
            random.shuffle(item_list)
        # Remove time
        item_list = list(map(lambda x: x[1], item_list))
        # TODO: Handle duplicated items
        # Split item
        train_item = item_list[int(len(item_list)*test_size):]
        test_item = item_list[:int(len(item_list)*test_size)]
        # Remove time
        train_item = set(train_item)
        test_item = set(test_item)

        assert len(test_item) > 0, "No test item for user %d" % user
        train_user_list[user] = train_item
        test_user_list[user] = test_item
    return train_user_list, test_user_list


def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair


def main(args):
    if args.dataset == 'ml-1m':
        df = MovieLens1M(args.data_dir).load()
    elif args.dataset == 'ml-20m':
        df = MovieLens20M(args.data_dir).load()
    elif args.dataset == 'gowalla':
        df = Gowalla(args.data_dir).load()
    else:
        raise NotImplementedError
    df = convert_unique_idx(df, 'user')
    df = convert_unique_idx(df, 'item')
    print('Complete assigning unique index to user and item')

    user_size = len(df['user'].unique())
    item_size = len(df['item'].unique())

    total_user_list = create_user_list(df, user_size)
    train_user_list, test_user_list = split_train_test(total_user_list,
                                                       test_size=args.test_size,
                                                       time_order=args.time_order)
    print('Complete spliting items for training and testing')

    train_pair = create_pair(train_user_list)
    print('Complete creating pair')

    dataset = {'user_size': user_size, 'item_size': item_size, 
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'train_pair': train_pair}
    dirname = os.path.dirname(os.path.abspath(args.output_data))
    os.makedirs(dirname, exist_ok=True)
    with open(args.output_data, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['ml-1m', 'ml-20m', 'gowalla'])
    parser.add_argument('--data_dir',
                        type=str,
                        default=os.path.join('data', 'ml-1m'),
                        help="File path for raw data")
    parser.add_argument('--output_data',
                        type=str,
                        default=os.path.join('preprocessed', 'ml-1m.pickle'),
                        help="File path for preprocessed data")
    parser.add_argument('--test_size',
                        type=float,
                        default=0.2,
                        help="Proportion for training and testing split")
    parser.add_argument('--time_order',
                        action='store_true',
                        help="Proportion for training and testing split")
    args = parser.parse_args()
    main(args)
