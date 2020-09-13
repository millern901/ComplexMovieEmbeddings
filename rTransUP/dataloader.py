from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, triples, arg_counts, negative_sample_size, is_rec, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)

        self.nuser = arg_counts[0]
        self.nrating = arg_counts[1]
        self.nitem = arg_counts[2]
        self.nrelation = arg_counts[3]
        self.nentity = arg_counts[4]

        self.item_range = arg_counts[2] if is_rec else arg_counts[4]
        self.is_rec = is_rec
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.negative_sample_size = negative_sample_size
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # load train triples
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        # calculate subsampling weight from triple frequency
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        # generate negative training sample
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            # create mask for negative kg train sample based on current training mode
            negative_sample = np.random.randint(self.item_range, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)

            # update negative sample list
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        # set positive and negative samples
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.is_rec, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        is_rec = data[0][3]
        mode = data[0][4]

        return positive_sample, negative_sample, subsample_weight, is_rec, mode

    @staticmethod
    def count_frequency(triples, start=4):
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, arg_counts, is_rec, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples

        self.nuser = arg_counts[0]
        self.nrating = arg_counts[1]
        self.nitem = arg_counts[2]
        self.nrelation = arg_counts[3]
        self.nentity = arg_counts[4]

        self.item_range = arg_counts[2] if is_rec else arg_counts[4]
        self.is_rec = is_rec
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # load triples and set tail/head range
        head, relation, tail = self.triples[idx]

        # create random test batch based on current test mode
        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.item_range)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.item_range)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        # generate filter bias from randomly generate test set
        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()

        # set negative and positive test samples
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.is_rec, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        is_rec = data[0][3]
        mode = data[0][4]

        return positive_sample, negative_sample, filter_bias, is_rec, mode


class TriDirectionalOneShotIterator(object):
    def __init__(self, rating_dataloader_tail, movie_dataloader_head, movie_dataloader_tail):
        self.rating_dataloader_tail = self.one_shot_iterator(rating_dataloader_tail)
        self.movie_dataloader_head = self.one_shot_iterator(movie_dataloader_head)
        self.movie_dataloader_tail = self.one_shot_iterator(movie_dataloader_tail)
        self.dataloader_list = [self.rating_dataloader_tail, self.movie_dataloader_head, self.movie_dataloader_tail]
        self.step = 0

    def __next__(self):
        data = next(self.dataloader_list[self.step % 3])
        self.step += 1
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
