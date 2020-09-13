from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TestDataset


class rTransUP(nn.Module):
    def __init__(self, arg_counts, hidden_dim, gamma):
        super(rTransUP, self).__init__()
        # Model element counts
        self.nuser = arg_counts[0]
        self.nrating = arg_counts[1]
        self.nitem = arg_counts[2]
        self.nrelation = arg_counts[3]
        self.nentity = arg_counts[4]

        # Model embedding Dimensions
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim * 2
        self.relation_dim = hidden_dim

        # Model constants
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        # Model embedding range
        self.epsilon = 2.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        # Model embedding sizes list
        self.sizes = [(self.nuser, self.entity_dim),
                      (self.nitem, self.entity_dim),
                      (self.nrelation, self.relation_dim),
                      (self.nentity, self.entity_dim),
                      (self.nrelation, self.relation_dim)]

        # Model embeddings list: [user(0), item(1), preference(2), entity(3), relation(4)]
        self.embeddings = nn.ParameterList([nn.Parameter(torch.zeros(s)) for s in self.sizes])

        # Model embeddings weight's initialization
        self.embeddings.apply(self._init_weights)

    def _init_weights(self, parameters):
        for parameter in parameters:
            nn.init.uniform_(
                tensor=parameter,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

    def forward(self, sample, is_rec, mode):
        # Recommendation forward
        if is_rec:
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                user = torch.index_select(
                    self.embeddings[0],
                    dim=0,
                    index=sample[:, 0]
                ).unsqueeze(1)
                item = torch.index_select(
                    self.embeddings[1],
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)
                entity = torch.index_select(
                    self.embeddings[3],
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)
            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                user = torch.index_select(
                    self.embeddings[0],
                    dim=0,
                    index=head_part[:, 0]
                ).unsqueeze(1)
                item = torch.index_select(
                    self.embeddings[1],
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
                entity = torch.index_select(
                    self.embeddings[3],
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)
            else:
                raise ValueError('mode %s not supported' % mode)
            score = self.complex_recommendation(user, item, entity, mode)
        # RotatE forward
        else:
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                head = torch.index_select(
                    self.embeddings[3],
                    dim=0,
                    index=sample[:, 0]
                ).unsqueeze(1)
                relation = torch.index_select(
                    self.embeddings[4],
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)
                tail = torch.index_select(
                    self.embeddings[3],
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)
            elif mode == 'head-batch':
                tail_part, head_part = sample
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
                head = torch.index_select(
                    self.embeddings[3],
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
                relation = torch.index_select(
                    self.embeddings[4],
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)
                tail = torch.index_select(
                    self.embeddings[3],
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head = torch.index_select(
                    self.embeddings[3],
                    dim=0,
                    index=head_part[:, 0]
                ).unsqueeze(1)
                relation = torch.index_select(
                    self.embeddings[4],
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)
                tail = torch.index_select(
                    self.embeddings[3],
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
            else:
                raise ValueError('mode %s not supported' % mode)
            score = self.rotate(head, relation, tail, mode)
        return score

    def rotate(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        # chunk the head and tail embeddings into real and complex parts
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # calculate relation phase and get real and imaginary parts
        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # calculate the real and imaginary scores
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        # calculate the norm of the scores
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)

        return score

    def complex_recommendation(self, user, item, entity, mode):
        # separate real and imaginary embeddings
        re_user, im_user = torch.chunk(user, 2, dim=2)
        re_entity, im_entity = torch.chunk(entity, 2, dim=2)
        re_item, im_item = torch.chunk(item, 2, dim=2)

        # calculate real and imaginary sums
        re_object = (re_user + re_entity) + re_item
        im_object = (im_user + im_entity) + im_item

        # calculate relation phase
        pi = 3.14159265358979323846
        relation_phase = (self.embeddings[2] + self.embeddings[4]) / (self.embedding_range.item() / pi)

        # calculate preference probabilities
        re_pref_prob = torch.matmul(re_object, torch.t(relation_phase))
        im_pref_prob = torch.matmul(im_object, torch.t(relation_phase))

        # calculate preference distribution sample
        re_pref_dist = F.gumbel_softmax(logits=re_pref_prob, hard=True, dim=2)
        im_pref_dist = F.gumbel_softmax(logits=im_pref_prob, hard=True, dim=2)

        # calculate preference embeddings
        re_preference = torch.matmul(re_pref_dist, relation_phase)
        im_preference = torch.matmul(im_pref_dist, relation_phase)

        # calculate final phases
        preference_phase = torch.atan2(im_preference, re_preference)
        head_phase = torch.atan2(im_user, re_user)
        tail_phase = torch.atan2(im_entity + im_item, re_entity + re_item)

        # calculate model score
        score = (head_phase + preference_phase) - tail_phase
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        # ready ready model for training and zero out all gradients
        model.train()
        optimizer.zero_grad()

        # grab the positive samples, negative samples and subsampling weight
        positive_sample, negative_sample, subsampling_weight, is_rec, mode = next(train_iterator)

        # set tensors to cuda
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # calculate the negative score
        negative_score = model(sample=(positive_sample, negative_sample), is_rec=is_rec, mode=mode)
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            # In normal sampling, we apply back-propagation on the sampling weight
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        # calculate the positive score
        positive_score = model(sample=positive_sample, is_rec=is_rec, mode='single')
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        # calculate positive and negative losses
        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        # average losses and back-propagate
        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()

        # create log file for training step and return it
        regularization_log = {}
        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, is_rec):
        # prepare model for evaluation
        model.eval()
        arg_counts = [args.nuser, args.nrating, args.nitem, args.nrelation, args.nentity]

        # prepare dataloader for Recommendation evaluation
        if is_rec:
            test_dataloader_tail = DataLoader(
                TestDataset(test_triples, all_true_triples, arg_counts, is_rec, 'tail-batch'),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )
        # prepare dataloader for RotatE evaluation
        else:
            test_dataloader_head = DataLoader(
                TestDataset(test_triples, all_true_triples, arg_counts, is_rec, 'head-batch'),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )
            test_dataloader_tail = DataLoader(
                TestDataset(test_triples, all_true_triples, arg_counts, is_rec, 'tail-batch'),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

        # initialize testing and log lists
        test_dataset_list = [test_dataloader_tail] if is_rec else [test_dataloader_head, test_dataloader_tail]
        logs = []

        # set initial step and total steps
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, is_rec, mode in dataset:
                    # set tensors to cuda
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    # grab batch size
                    batch_size = positive_sample.size(0)
                    is_eval = True if batch_size <= 16 else False

                    # score the model
                    score = model(sample=(positive_sample, negative_sample), is_rec=is_rec, mode=mode)
                    score += filter_bias

                    if is_eval and is_rec:
                        print(score)

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)
                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        if is_eval and is_rec:
                            print(ranking)
                            print('\n')

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })
                    if is_eval and is_rec:
                        print('\n\n\n')
                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        # average metrics
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
