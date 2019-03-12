#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
# Few methods have been adapted from https://github.com/facebookresearch/DrQA
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data processing/loading helpers."""


import numpy as np
import json
import logging
from smart_open import smart_open
import unicodedata
import heapq
import os

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .vector import vectorize

logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self, args):
        self.args = args
        if not args.create_vocab:
            logger.info('[ Reading vocab files from {}]'.format(args.vocab_dir))
            self.tok2ind = json.load(open(args.vocab_dir+'tok2ind.json'))
            self.ind2tok = json.load(open(args.vocab_dir+'ind2tok.json'))

        else:
            self.tok2ind = {self.NULL: 0, self.UNK: 1}
            self.ind2tok = {0: self.NULL, 1: self.UNK}
            self.oov_words = {}

            # Index words in embedding file
            if args.pretrained_words and args.embedding_file:
                logger.info('[ Indexing words in embedding file... ]')
                self.valid_words = set()
                with smart_open(args.embedding_file) as f:
                    for line in f:
                        w = self.normalize(line.decode('utf-8').rstrip().split(' ')[0])
                        self.valid_words.add(w)
                logger.info('[ Num words in set = %d ]' % len(self.valid_words))
            else:
                self.valid_words = None

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def add(self, token):
        token = self.normalize(token)
        if self.valid_words and token not in self.valid_words:
            # logger.info('{} not a valid word'.format(token))
            if token not in self.oov_words:
                self.oov_words[token] = len(self.oov_words)
            return
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def swap_top(self, top_words):
        """
        Reindexes the dictionary to have top_words labelled 2:N.
        (0, 1 are for <NULL>, <UNK>)
        """
        for idx, w in enumerate(top_words, 2):
            if w in self.tok2ind:
                w_2, idx_2 = self.ind2tok[idx], self.tok2ind[w]
                self.tok2ind[w], self.ind2tok[idx] = idx, w
                self.tok2ind[w_2], self.ind2tok[idx_2] = idx_2, w_2

    def save(self):

        fout = open(os.path.join(self.args.vocab_dir, "ind2tok.json"), "w")
        json.dump(self.ind2tok, fout)
        fout.close()
        fout = open(os.path.join(self.args.vocab_dir, "tok2ind.json"), "w")
        json.dump(self.tok2ind, fout)
        fout.close()
        logger.info("Dictionary saved at {}".format(self.args.vocab_dir))


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class SquadDataset(Dataset):
    def __init__(self, args, examples, word_dict,
                 feature_dict, single_answer=False, para_mode=False, train_time=True):
        self.examples = examples
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.args = args
        self.single_answer = single_answer
        self.para_mode = para_mode
        self.train_time = train_time

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.args, self.examples[index], self.word_dict, self.feature_dict, self.single_answer,
                         self.para_mode, self.train_time)

    def lengths(self):
        if not self.para_mode:
            return [(len(ex['document']), len(ex['question'])) for ex in self.examples]
        else:
            q_key = 'question_str' if (self.args.src == 'triviaqa' or self.args.src == 'qangaroo') else 'question'
            return [(len(ex['document']), max([len(para) for para in ex['document']]), len(ex[q_key])) for ex in self.examples]

class MultiCorpusDataset(Dataset):
    def __init__(self, args, corpus, word_dict,
                 feature_dict, single_answer=False, para_mode=True, train_time=True):
        self.corpus = corpus
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.args = args
        self.single_answer = single_answer
        self.para_mode = para_mode
        self.train_time = train_time
        self.pid_list = list(self.corpus.paragraphs.keys())
    def __len__(self):
        if self.para_mode:
            return len(self.corpus.paragraphs)
        else:
            return len(self.corpus.questions)

    def __getitem__(self, index):
        if self.para_mode:
            ex = {}
            pid =  self.pid_list[index]
            para = self.corpus.paragraphs[pid]
            assert pid == para.pid
            ex['document'] = para.text
            ex['id'] = para.pid
            ex['ans_occurance'] = para.ans_occurance
            qid = para.qid
            question = self.corpus.questions[qid]
            ex['question'] = question.text
            assert pid in question.pids

            return vectorize(self.args, ex)
        else:
            raise NotImplementedError("later")


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------

class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, para_mode=False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.para_mode = para_mode

    def __iter__(self):
        if not self.para_mode:
            lengths = np.array(
                [(-l[0], -l[1], np.random.random()) for l in self.lengths],
                dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
            )
        else:
            lengths = np.array([(-l[0], -l[1], -l[2], np.random.random()) for l in self.lengths], dtype=[('l1', np.int_), ('l2', np.int_), ('l3', np.int_), ('rand', np.float_)])
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand')) if not self.para_mode else np.argsort(lengths, order=('l1', 'l2', 'l3', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)

class CorrectParaSortedBatchSampler(Sampler):
    """
    This awesome sampler was written by Peng Qi (http://qipeng.me/)
    """
    def __init__(self, dataset, batch_size, shuffle=True, para_mode=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.para_mode = para_mode

    def __iter__(self):
        import sys
        correct_paras = [(ex[5] > 0).long().sum() for ex in self.dataset]

        # make sure the number of correct paras in each minibatch is about the same
        mean = sum(correct_paras) / len(correct_paras)
        target = mean * self.batch_size

        # also make sure the number of total paras in each minibatch is about the same
        lengths = [x[0] for x in self.dataset.lengths()]
        target2 = sum(lengths) / len(lengths) * self.batch_size

        heuristic_weight = 0.1 # heuristic importance of making sum_para_len uniform compared to making sum_correct_paras uniform

        indices = [x[0] for x in sorted(enumerate(zip(correct_paras, lengths)), key=lambda x: x[1], reverse=True)]

        batches = [[] for _ in range((len(self.dataset) + self.batch_size - 1) // self.batch_size)]

        batches_by_size = {0: {0: [(i, 0, 0) for i in range(len(batches))] } }

        K = 100 # "beam" size

        for idx in indices:
            costs = []
            for size in batches_by_size:
                cost_reduction = -(2 * size + correct_paras[idx] - 2 * target) * correct_paras[idx]

                costs += [(size, cost_reduction)]

            costs = heapq.nlargest(K, costs, key=lambda x: x[1])

            best_cand = None
            for size, cost in costs:
                best_size2 = -1
                best_reduction = -float('inf')
                for size2 in batches_by_size[size]:
                    cost_reduction = -(2 * size2 + lengths[idx] - 2 * target2) * lengths[idx]

                    if cost_reduction > best_reduction:
                        best_size2 = size2
                        best_reduction = cost_reduction

                assert best_size2 >= 0

                cost_reduction_all = cost + best_reduction * heuristic_weight
                if best_cand is None or cost_reduction_all > best_cand[2]:
                    best_cand = (size, best_size2, cost_reduction_all, cost, best_reduction)

            assert best_cand is not None

            best_size, best_size2 = best_cand[:2]

            assert len(batches_by_size[best_size]) > 0

            # all batches of the same size are created equal
            best_batch, batches_by_size[best_size][best_size2] = batches_by_size[best_size][best_size2][0], batches_by_size[best_size][best_size2][1:]

            if len(batches_by_size[best_size][best_size2]) == 0:
                del batches_by_size[best_size][best_size2]
                if len(batches_by_size[best_size]) == 0:
                    del batches_by_size[best_size]

            batches[best_batch[0]] += [idx]
            newsize = best_batch[1] + correct_paras[idx]
            newsize2 = best_batch[2] + lengths[idx]

            if len(batches[best_batch[0]]) < self.batch_size:
                # insert back
                if newsize not in batches_by_size:
                    batches_by_size[newsize] = {}

                if newsize2 not in batches_by_size[newsize]:
                    batches_by_size[newsize][newsize2] = [(best_batch[0], newsize, newsize2)]
                else:
                    batches_by_size[newsize][newsize2] += [(best_batch[0], newsize, newsize2)]

        if self.shuffle:
            np.random.shuffle(batches)

        return iter([x for batch in batches for x in batch])

    def __len__(self):
        return len(self.dataset)