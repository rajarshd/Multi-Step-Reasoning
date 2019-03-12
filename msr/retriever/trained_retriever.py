import json
import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable

logger = logging.getLogger()


class Retriever(object):
    def __init__(self, args, read_dir):
        self.args = args
        self.read_dir = read_dir
        self.qid2filemap = json.load(open(os.path.join(read_dir, "map.json")))
        self.reverse_qid2filemap = {v: k for k, v in self.qid2filemap.items()}
        self.qid2correctparamap = json.load(open(os.path.join(read_dir, "correct_paras.json")))
        self.qids = None
        self.current_query_vectors = None  # stores the current value of query vectors for the batch
        self.current_para_vectors = None  # all para vectors
        self.all_query_vectors = None
        self.all_para_vectors = None
        self.cum_num_paras = []
        self.all_cum_num_paras = []
        self.embedding_dim = 128
        logger.info("Reading saved paragraph and query vectors from disk...{}".format(self.read_dir))
        self.all_query_vectors = np.load(os.path.join(read_dir, "question.npy"))
        self.all_query_vectors = torch.FloatTensor(self.all_query_vectors)
        self.all_para_vectors = np.load(os.path.join(read_dir, "document.npy"))
        self.all_para_vectors = torch.FloatTensor(self.all_para_vectors)
        self.all_cum_num_paras = np.load(os.path.join(read_dir, "all_cumlen.npy"))
        self.all_cum_num_paras = torch.LongTensor(self.all_cum_num_paras)
        # self.qid2indexmap = torch.load(os.path.join(read_dir,  "qid2indexmap.pkl"))

        # test cases
        assert self.all_cum_num_paras.size(0) == self.all_query_vectors.size(0)
        assert self.all_cum_num_paras[-1] == self.all_para_vectors.size(0)
        logger.info("Done Reading!")

    def reset(self):
        """
        resets for a new batch of queries
        :return:
        """
        self.current_query_vectors = None
        self.current_para_vectors = None
        self.cum_num_paras = []

    def __call__(self, qids, train_time=False):

        # transform qids from strings to int
        self.qqids = qids
        self.qids = [self.qid2filemap[qid] for qid in qids]
        if self.current_query_vectors is None:  # first time; read from disk


            for i, qid in enumerate(self.qids):
                en_ind = self.all_cum_num_paras[qid]
                st_ind = 0 if qid == 0 else self.all_cum_num_paras[qid - 1]
                self.current_para_vectors = self.all_para_vectors[
                                            st_ind:en_ind] if self.current_para_vectors is None else torch.cat(
                    [self.current_para_vectors, self.all_para_vectors[st_ind:en_ind]], dim=0)

                self.cum_num_paras.append(self.current_para_vectors.size(0))

            self.current_query_vectors = torch.index_select(self.all_query_vectors, 0, torch.LongTensor(self.qids))
            if self.args.cuda:
                self.current_query_vectors = Variable(self.current_query_vectors.cuda())
                self.current_para_vectors = Variable(self.current_para_vectors.cuda())

        # take inner product
        para_scores = torch.mm(self.current_query_vectors, self.current_para_vectors.t())
        # now for each query, slice out the scores for the corresponding paras
        sorted_para_ids_per_query = []
        sorted_para_scores_per_query = []
        all_num_positive_paras = []
        for i in range(para_scores.size(0)):  # for each query
            st = 0 if i == 0 else self.cum_num_paras[i - 1]
            en = self.cum_num_paras[i]
            para_scores_for_query = para_scores[i, st:en]
            sorted_scores, para_ids_query = torch.sort(para_scores_for_query, descending=True)

            if train_time or self.args.cheat:
                # during train time, make sure that the top (may be few) paras have annotation
                # get correct_paras
                correct_para_ids = self.qid2correctparamap[self.reverse_qid2filemap[self.qids[i]]]
                # for some qids, there arent any labels, will have to handle them separately in model.update
                if len(correct_para_ids) > 0:
                    np.random.shuffle(correct_para_ids)
                    num_positive_paras = min(self.args.num_positive_paras, len(correct_para_ids))
                    correct_para_ids = correct_para_ids[:num_positive_paras]
                    para_ids_query = para_ids_query.cpu().data.numpy().tolist()
                    sorted_scores = sorted_scores.cpu().data.numpy().tolist()
                    temp_para_ids_query = []
                    temp_sorted_scores = []
                    correct_para_inds = []
                    for i, p in enumerate(para_ids_query):
                        if p not in correct_para_ids:
                            temp_para_ids_query.append(p)
                            temp_sorted_scores.append(sorted_scores[i])
                        else:
                            correct_para_inds.append(i)
                    para_ids_query = correct_para_ids + temp_para_ids_query
                    sorted_scores = [sorted_scores[i] for i in correct_para_inds] + temp_sorted_scores
                    para_ids_query = Variable(torch.LongTensor(para_ids_query))
                    sorted_scores = Variable(torch.FloatTensor(sorted_scores))
                    if self.args.cuda:
                        para_ids_query = para_ids_query.cuda()
                        sorted_scores = sorted_scores.cuda()

                    all_num_positive_paras.append(num_positive_paras)
                else:
                    all_num_positive_paras.append(0)

            sorted_para_ids_per_query.append(para_ids_query.data)
            sorted_para_scores_per_query.append(sorted_scores)

        return self.current_query_vectors, sorted_para_scores_per_query, sorted_para_ids_per_query, all_num_positive_paras

    def update_query_vectors(self, q_vectors):
        self.current_query_vectors = q_vectors

    def get_nearest_correct_para_vector(self):

        # gather the top para_id for each qid
        top_para_ids, incorrect_para_ids, mask = [], [], []

        for i, qid in enumerate(self.qqids):
            st = 0 if i == 0 else self.cum_num_paras[i - 1]
            correct_paras = self.qid2correctparamap[self.reverse_qid2filemap[self.qids[i]]]
            np.random.shuffle(correct_paras)
            try:
                top_para_ids.append(correct_paras[0] + st)
                mask.append(1)
            except IndexError:
                top_para_ids.append(0 + st)  # some question-paras have no answer occurrences
                mask.append(0)
            incorrect_paras = list(set(range(self.cum_num_paras[i] - st)) - set(correct_paras))
            np.random.shuffle(incorrect_paras)
            if len(incorrect_paras) == 0:
                incorrect_para_ids.append(0 + st)
                mask[i] = 0
            else:
                incorrect_para_ids.append(incorrect_paras[0] + st)
        # now select the appropriate para vector
        top_para_ids = torch.cuda.LongTensor(top_para_ids) if self.args.cuda else torch.LongTensor(top_para_ids)
        incorrect_para_ids = torch.cuda.LongTensor(incorrect_para_ids) if self.args.cuda else torch.LongTensor(
            incorrect_para_ids)
        mask = torch.cuda.ByteTensor(mask) if self.args.cuda else torch.ByteTensor(mask)
        nearest_correct_paras = torch.index_select(self.current_para_vectors, 0, Variable(top_para_ids))
        farthest_incorrect_paras = torch.index_select(self.current_para_vectors, 0, Variable(incorrect_para_ids))
        return nearest_correct_paras, farthest_incorrect_paras, mask



