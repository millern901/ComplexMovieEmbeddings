from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import argparse
import logging
import random
import os


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Creating Training, Testing, and Validation sets for User Data',
        usage='train.py [<args>]'
    )
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--relation', action='store_true')
    parser.add_argument('--filter_unseen', action='store_true')

    parser.add_argument('--frequency', default=10, type=int)
    parser.add_argument('--train_ratio', default=.7, type=float)
    parser.add_argument('--test_ratio', default=.2, type=float)

    return parser.parse_args(args)


def preprocess(data_path, file_path, map_file, ratios=None, frequency=10, relation=False, filter_unseen=False, logger=None):
    # create file paths
    train_file = os.path.join(file_path, "train.txt")
    test_file = os.path.join(file_path, "test.txt")
    valid_file = os.path.join(file_path, "valid.txt")
    user_map_file = os.path.join(file_path, "user.dict")
    item_map_file = os.path.join(file_path, "item.dict")
    rating_map_file = os.path.join(file_path, "rating.dict")

    # filter low frequency users and movies
    if logger is not None:
        logger.info("Filtering infrequent users and items (<={}) ..."
                    .format(frequency))
    user_dict = cutLowFrequentData(rating_file=data_path, mapping_file=map_file, frequency=frequency, relation=relation)

    # split model files
    if logger is not None:
        logger.info("Beginning the splitting of data into {:.2f}% training, {:.2f}% testing, {:.2f}% validation"
                    .format(ratios[0]*100, ratios[1]*100, ratios[2]*100))
    split_data = splitRatingData(user_dict=user_dict, ratios=ratios, filter_unseen=filter_unseen)

    # log results
    if logger is not None:
        logger.info("There are: {} train, {} test and {} validate examples!"
                    .format(len(split_data[0]), len(split_data[2]), len(split_data[1])))
        logger.info("There are: {} interactions with {} users and {} movies!"
                    .format(len(split_data[0]) + len(split_data[1]) + len(split_data[2]), len(split_data[3]), len(split_data[5])))

    # save dictionaries
    with open(user_map_file, 'w', encoding='utf-8') as fout:
        for i in range(len(split_data[3])):
            fout.write('{}\t{}\n'.format(i, split_data[3][i]))
    with open(item_map_file, 'w', encoding='utf-8') as fout:
        for i in range(len(split_data[5])):
            fout.write('{}\t{}\n'.format(i, split_data[5][i]))
    with open(rating_map_file, 'w', encoding='utf-8') as fout:
        for i in range(len(split_data[4])):
            fout.write('{}\t{}\n'.format(i, split_data[4][i]))

    # save model files
    with open(train_file, 'w', encoding='utf-8') as fout:
        for rating in split_data[0]:
            fout.write('{}\t{}\t{}\n'.format(rating[0], rating[1], rating[2]))
    with open(valid_file, 'w', encoding='utf-8') as fout:
        for rating in split_data[1]:
            fout.write('{}\t{}\t{}\n'.format(rating[0], rating[1], rating[2]))
    with open(test_file, 'w', encoding='utf-8') as fout:
        for rating in split_data[2]:
            fout.write('{}\t{}\t{}\n'.format(rating[0], rating[1], rating[2]))


def cutLowFrequentData(rating_file, mapping_file, frequency=10, relation=False):
    # load rating data frame
    rating_df = pd.read_csv(rating_file, header=None, sep='::', engine='python',
                            names=['userId', 'movieId', 'rating', 'timeslice'])
    rating_df = rating_df[['userId', 'rating', 'movieId']]

    # load DBpedia movie mappings
    movie_dict = dict()
    movie_list = []
    if mapping_file is not None and os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_split = line.strip().split('\t')
                if len(line_split) < 3:
                    continue
                movie_dict[str(line_split[0])] = line_split[2]
                movie_list.append(line_split[0])

    # drop movies with no movie data
    rating_df.drop(rating_df[~rating_df.movieId.isin(movie_list)].index, inplace=True)

    # remove low frequency users and movies
    rating_df = rating_df[rating_df.groupby('userId')['movieId'].transform('sum') > frequency]
    rating_df = rating_df[rating_df.groupby('movieId')['userId'].transform('sum') > frequency]

    # add entity and relation handles 
    rating_df['userId'] = rating_df.userId.apply(lambda x: 'user_' + str(x))
    rating_df['movieId'] = rating_df.movieId.apply(lambda x: movie_dict[str(x)])
    rating_df['rating'] = rating_df.rating.apply(lambda x: 'rating_' + str(x) if relation else 'has_watched')

    # create user dictionary
    temp = rating_df.values
    user_dict = dict()
    for line in temp:
        if line[0] in user_dict.keys():
            user_dict[line[0]].append((line[1], line[2]))
        else:
            user_dict[line[0]] = [(line[1], line[2])]

    # return updated rating data frame
    return user_dict


def splitRatingData(user_dict, ratios=None, filter_unseen=False):
    # initialize all sets and temporary lists
    user_set = set()
    rating_set = set()
    movie_set = set()
    train_item_set = set()
    tmp_train_list = []
    tmp_valid_list = []
    tmp_test_list = []

    # split each user into train, valid, and test lists
    for user in user_dict.keys():
        # add user to set
        user_set.add(user)

        # create splitting indices
        tmp_item_list = user_dict[user]
        n_items = len(tmp_item_list)
        n_train = round(ratios[0] * n_items)
        n_test = round(n_train + ratios[1] * n_items)

        # extend temporary train/test/validate lists and and relations and movies to sets
        random.shuffle(tmp_item_list)
        for rm in tmp_item_list[:n_train]:
            tmp_train_list.append((user, rm[0], rm[1]))
            train_item_set.add(rm[1])
            rating_set.add(rm[0])
            movie_set.add(rm[1])
        for rm in tmp_item_list[n_train:n_test]:
            tmp_test_list.append((user, rm[0], rm[1]))
            rating_set.add(rm[0])
            movie_set.add(rm[1])
        for rm in tmp_item_list[n_test:]:
            tmp_valid_list.append((user, rm[0], rm[1]))
            rating_set.add(rm[0])
            movie_set.add(rm[1])

    # create final train list
    train_list = [(rating[0], rating[1], rating[2]) for rating in tmp_train_list]

    # filter unseen samples
    if filter_unseen:
        # create final filtered test and valid lists
        test_list = [(rating[0], rating[1], rating[2]) for rating in tmp_test_list if rating[2] in train_item_set]
        valid_list = [(rating[0], rating[1], rating[2]) for rating in tmp_valid_list if rating[2] in train_item_set]
    else:
        # create final filtered test and valid lists
        test_list = [(rating[0], rating[1], rating[2]) for rating in tmp_test_list]
        valid_list = [(rating[0], rating[1], rating[2]) for rating in tmp_valid_list]

    return train_list, valid_list, test_list, list(user_set), list(rating_set), list(movie_set)


def main(args):
    # error handling
    if args.data_path is None:
        raise ValueError('Data path must be chosen.')
    if args.dataset is None:
        raise ValueError('Dataset must be chosen.')
    if 0 > args.frequency:
        raise ValueError('Minimum frequency is out of range.')
    if 0 > args.train_ratio or args.train_ratio > 1:
        raise ValueError('Train ratio is out of range.')
    if 0 > args.test_ratio or args.test_ratio > 1:
        raise ValueError('Test ratio is out of range.')
    valid_ratio = 1 - args.train_ratio - args.test_ratio
    if 0 > valid_ratio or valid_ratio > 1:
        raise ValueError('Valid ratio is out of range.')
    ratios = [args.train_ratio, args.test_ratio, valid_ratio]

    # initialize file paths
    in_head_path = os.path.join(args.data_path, 'source')
    out_head_path = os.path.join(args.data_path, 'preprocess')
    # in paths
    in_path = os.path.join(in_head_path, args.dataset)
    data_in_path = os.path.join(in_path, 'ratings.dat')
    mapping_file_path = os.path.join(in_path, 'MappingMovielens2DBpedia-1.2.tsv')
    # out path
    out_path = os.path.join(out_head_path, args.dataset)
    data_out_path = os.path.join(out_path, 'rating')

    # initialize logger
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    log_file = os.path.join(out_path, "rating_data_preprocess.log")
    # logger format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # run rating preprocess
    preprocess(data_path=data_in_path, file_path=data_out_path, map_file=mapping_file_path, ratios=ratios,
               frequency=args.frequency, relation=args.relation, filter_unseen=args.filter_unseen, logger=logger)


if __name__ == '__main__':
    main(parse_args())



