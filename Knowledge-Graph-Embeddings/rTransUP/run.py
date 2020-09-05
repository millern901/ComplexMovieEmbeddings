from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader

from model import rTransUP
from dataloader import TrainDataset, TriDirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--data_path', type=str, default=None)

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nuser', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nitem', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrating', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)


def override_config(args):
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    if args.data_path is None:
        args.data_path = argparse_dict['data_path']

    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    # save all model embeddings
    embeddings = model.embeddings
    file_names = ['user_embedding', 'item_embedding', 'preference_embedding', 'entity_embedding', 'relation_embedding']
    for i in range(len(embeddings)):
        embedding = embeddings[i].detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, file_names[i]),
            embedding
        )


def read_rating_triple(file_path, user2id, rating2id, item2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((user2id[h], rating2id[r], item2id[t]))
    return triples


def read_movie_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            # triples.append((entity2id[h], relation2id[r], entity2id[t]))
            triples.append((entity2id[h], relation2id[t], entity2id[r]))
    return triples


def set_logger(args):
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    # model argument error handling and logger initialization
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be chosen.')
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be chosen.')
    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_logger(args)

    # load variable dictionaries
    with open(os.path.join(args.data_path, 'rating/user.dict')) as fin:
        u2id = dict()
        for line in fin:
            uid, user = line.strip().split('\t')
            u2id[user] = int(uid)
    with open(os.path.join(args.data_path, 'rating/item.dict')) as fin:
        i2id = dict()
        for line in fin:
            iid, item = line.strip().split('\t')
            i2id[item] = int(iid)
    with open(os.path.join(args.data_path, 'rating/rating.dict')) as fin:
        rate2id = dict()
        for line in fin:
            rid, rating = line.strip().split('\t')
            rate2id[rating] = int(rid)
    with open(os.path.join(args.data_path, 'kg/entities.dict')) as fin:
        e2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            e2id[entity] = int(eid)
    with open(os.path.join(args.data_path, 'kg/relations.dict')) as fin:
        r2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            r2id[relation] = int(rid)

    # set counts
    args.nuser = nuser = len(u2id)
    args.nitem = nitem = len(i2id)
    args.nrating = nrating = len(rate2id)
    args.nentity = nentity = len(e2id)
    args.nrelation = nrelation = len(r2id)
    arg_counts = [nuser, nrating, nitem, nrelation, nentity]

    # log model initialization parameters
    logging.info('Version: 1.0')
    logging.info('Model: rTransUP')
    logging.info('Data Path: %s' % args.data_path)
    logging.info('Number of users: %d' % nuser)
    logging.info('Number of ratings: %d' % nrating)
    logging.info('Number of items: %d' % nitem)
    logging.info('Number of entities: %d' % nentity)
    logging.info('Number of relations: %d' % nrelation)

    # load all rating and movie triples
    ratings_train_triples = read_rating_triple(os.path.join(args.data_path, 'rating/train.txt'), u2id, rate2id, i2id)
    ratings_valid_triples = read_rating_triple(os.path.join(args.data_path, 'rating/valid.txt'), u2id, rate2id, i2id)
    ratings_test_triples = read_rating_triple(os.path.join(args.data_path, 'rating/test.txt'), u2id, rate2id, i2id)
    movie_train_triples = read_movie_triple(os.path.join(args.data_path, 'kg/train.txt'), e2id, r2id)
    movie_valid_triples = read_movie_triple(os.path.join(args.data_path, 'kg/valid.txt'), e2id, r2id)
    movie_test_triples = read_movie_triple(os.path.join(args.data_path, 'kg/test.txt'), e2id, r2id)

    # log triple lengths
    logging.info('Number of rating train triples: %d' % len(ratings_train_triples))
    logging.info('Number of rating valid triples: %d' % len(ratings_valid_triples))
    logging.info('Number of rating test triples: %d' % len(ratings_test_triples))
    logging.info('Number of movie train triples: %d' % len(movie_train_triples))
    logging.info('Number of movie valid triples: %d' % len(movie_valid_triples))
    logging.info('Number of movie test triples: %d' % len(movie_test_triples))

    # collect all true rating and movie triples
    all_true_rating_triples = ratings_train_triples + ratings_valid_triples + ratings_test_triples
    all_true_movie_triples = movie_train_triples + movie_valid_triples + movie_test_triples

    # initialize model parameters
    rtransup_model = rTransUP(
        arg_counts=arg_counts,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma
    )

    # log model parameters
    logging.info('Model Parameter Configuration:')
    param_number = 0
    param_names = ['Gamma', 'Embedding Range', 'User Embeddings', 'Item Embeddings',
                   'Preference Embeddings', 'Entity Embeddings', 'Relation Embeddings']
    for name, param in rtransup_model.named_parameters():
        logging.info('Parameter %s: Size = %s, Requires Grad = %s'
                     % (param_names[param_number], str(param.size()), str(param.requires_grad)))
        param_number += 1

    # set model to gpu
    if args.cuda:
        rtransup_model = rtransup_model.cuda()

    if args.do_train:
        # initialize training dataloader
        rating_loader_tail = DataLoader(
            TrainDataset(ratings_train_triples, arg_counts, args.negative_sample_size, True, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        movie_loader_head = DataLoader(
            TrainDataset(movie_train_triples, arg_counts, args.negative_sample_size, False, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        movie_loader_tail = DataLoader(
            TrainDataset(movie_train_triples, arg_counts, args.negative_sample_size, False, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        train_iterator = TriDirectionalOneShotIterator(rating_loader_tail, movie_loader_head, movie_loader_tail)

        # initialize training optimizer
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, rtransup_model.parameters()),
            lr=current_learning_rate
        )

        # initialize warm up steps
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        rtransup_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing rTransUP Model...')
        init_step = 0
    step = init_step

    # log beginning of training
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)
        training_logs = []

        # training Loop
        for step in range(init_step, args.max_steps):
            log = rtransup_model.train_step(rtransup_model, optimizer, train_iterator, args)
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, rtransup_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(rtransup_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                # Recommendation model
                metrics = rtransup_model.test_step(rtransup_model, ratings_valid_triples,
                                                   all_true_rating_triples, args, True)
                log_metrics('Recommendation Valid', step, metrics)
                # RotatE model
                metrics = rtransup_model.test_step(rtransup_model, movie_valid_triples,
                                                   all_true_movie_triples, args, False)
                log_metrics('RotatE Valid', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(rtransup_model, optimizer, save_variable_list, args)

    if args.do_valid:
        # Recommendation model
        metrics = rtransup_model.test_step(rtransup_model, ratings_valid_triples,
                                           all_true_rating_triples, args, True)
        log_metrics('Recommendation Valid', step, metrics)
        # RotatE model
        metrics = rtransup_model.test_step(rtransup_model, movie_valid_triples,
                                           all_true_movie_triples, args, False)
        log_metrics('RotatE Valid', step, metrics)

    if args.do_test:
        # Recommendation model
        metrics = rtransup_model.test_step(rtransup_model, ratings_test_triples,
                                           all_true_rating_triples, args, True)
        log_metrics('Recommendation Test', step, metrics)
        # RotatE model
        metrics = rtransup_model.test_step(rtransup_model, movie_test_triples,
                                           all_true_movie_triples, args, False)
        log_metrics('RotatE Test', step, metrics)

    if args.evaluate_train:
        # Recommendation model
        metrics = rtransup_model.test_step(rtransup_model, ratings_train_triples,
                                           all_true_rating_triples, args, True)
        log_metrics('Recommendation Test', step, metrics)
        # RotatE model
        metrics = rtransup_model.test_step(rtransup_model, movie_train_triples,
                                           all_true_movie_triples, args, False)
        log_metrics('RotatE Train', step, metrics)


if __name__ == '__main__':
    main(parse_args())
