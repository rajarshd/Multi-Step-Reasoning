#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA Document Reader model"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy
import os

from torch.autograd import Variable
from .config import override_model_args
from .rnn_reader import RnnDocReader, MultiStepReasoner
from .utils import logsumexp
from msr.reader import utils


from . import layers
from collections import defaultdict

from msr.retriever.trained_retriever import Retriever

logger = logging.getLogger(__name__)



class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, word_dict, feature_dict,
                 state_dict=None, multi_step_reasoner_state_dict=None, multi_step_reader_state_dict=None,
                 multi_step_reader_self_attn_state_dict=None, normalize=True):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.env = Environment(args)

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

        # retriever
        self.ret = Retriever(args, read_dir=os.path.join(args.saved_para_vectors_dir, "train"))
        self.dev_ret = Retriever(args, read_dir=os.path.join(args.saved_para_vectors_dir,
                                                             "test")) if self.args.test else Retriever(args,
                                                                                                       read_dir=os.path.join(
                                                                                                           args.saved_para_vectors_dir,
                                                                                                           "dev"))
        # multi-step reader
        self.multi_step_reader = MultiStepReasoner(args, 2 * args.doc_layers * args.hidden_size,
                                                   2 * args.doc_layers * args.hidden_size)
        # multi-step reasoner
        self.multi_step_reasoner = MultiStepReasoner(args, args.doc_hidden_size, 600)
        self.reader_self_attn = layers.LinearSeqAttn(args.doc_hidden_size)
        if self.args.cuda:
            self.reader_self_attn = self.reader_self_attn.cuda()
        if multi_step_reasoner_state_dict:
            self.multi_step_reasoner.load_state_dict(multi_step_reasoner_state_dict)
        if multi_step_reader_state_dict:
            self.multi_step_reader.load_state_dict(multi_step_reader_state_dict)
        if multi_step_reader_self_attn_state_dict:
            self.reader_self_attn.load_state_dict(multi_step_reader_self_attn_state_dict)

        if self.args.freeze_reader:
            logger.info("Freezing the reader...")
            for params in self.network.parameters():
                params.requires_grad = False
            if self.multi_step_reader:
                for params in self.multi_step_reader.parameters():
                    params.requires_grad = False


        self.args.cuda = True


    def expand_dictionary(self, words):
        """Add words to the Model dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def load_embeddings(self, args, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        if args.embedding_table:
            logger.info('Loading embeddings from saved embeddings table at {}'.format(args.embedding_table_path))
            embeddings = torch.load(args.embedding_table_path)
            self.network.embedding.weight.data = embeddings
            logger.info('Loaded embeddings for {} words'.format(embeddings.size(0)))
            return


        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)
        torch.save(embedding, args.embedding_table_path)
        logger.info("Embedding table saved at {}".format(args.embedding_table_path))
        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.multi_step_reasoner is not None:
            parameters += [p for p in self.multi_step_reasoner.parameters() if p.requires_grad]
            parameters += [p for p in self.reader_self_attn.parameters() if p.requires_grad]

        if self.multi_step_reader is not None:
            parameters += [p for p in self.multi_step_reader.parameters() if p.requires_grad]

        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex, epoch_counter=0, ground_truths_map=None):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()
        loss, multi_step_loss, rl_loss = 0.0, 0.0, 0.0
        flag = 0
        for t in range(self.args.multi_step_reasoning_steps):
            qids = ex[-1]
            # 1. get query vector, para scores and para_ids
            query_vectors, sorted_para_scores_per_query, sorted_para_ids_per_query, all_num_positive_paras = self.ret(
                qids, train_time=not self.args.freeze_reader)
            if self.args.freeze_reader:
                para_ids = []
                filtered_q_counter = []
                for q_counter, ranked_para_ids in enumerate(sorted_para_ids_per_query):
                    if len(ranked_para_ids) <= self.args.num_positive_paras:
                        # include all paragraphs
                        para_ids.append(ranked_para_ids.cpu().tolist())
                        # since gather requires the same number of elements, repeat the top para, a bit hacky
                        para_ids[-1] += [ranked_para_ids[0] for _ in
                                         range(self.args.num_positive_paras - len(ranked_para_ids))]
                    else:
                        para_ids.append(ranked_para_ids[:self.args.num_positive_paras].tolist())
                    filtered_q_counter.append(q_counter)

                ind = torch.LongTensor(para_ids).unsqueeze(2)  # B X num_paras_test X 1
                ind = ind.expand(-1, -1, ex[0].size(2))  # B X num_paras_test X max_num_words
            else:
                # slice the tensors accordingly to select the top paragraph or top_para + some negative samples
                para_ids = []
                filtered_q_counter = []
                for q_counter, ranked_para_ids in enumerate(sorted_para_ids_per_query):
                    ranked_para_ids = ranked_para_ids.tolist()
                    para_ids_for_query = []
                    try:
                        para_ids_for_query += ranked_para_ids[
                                              :all_num_positive_paras[q_counter]]  # add (possibly multiple) top ranked para
                    except IndexError:
                        import pdb
                        pdb.set_trace()
                    if self.args.num_low_ranked_paras > 0:
                        # add negative samples (i.e. low ranked paragraphs)
                        # ignore train instances which dont have sufficient number of paras
                        if len(ranked_para_ids) < self.args.num_positive_paras + self.args.num_low_ranked_paras:
                            logger.debug("eliminated. Num paras = {}".format(len(ranked_para_ids)))
                            continue
                        para_ids_for_query += ranked_para_ids[len(ranked_para_ids)-self.args.num_low_ranked_paras:]
                    diff = (self.args.num_positive_paras + self.args.num_low_ranked_paras) - len(para_ids_for_query)
                    if diff > 0:
                        # this happened because the number of positive paragraphs sent by the ret is less than the
                        # self.args.num_positive_paras -- replace with random paras
                        # sample random paras
                        random_para_ids = np.random.choice(ranked_para_ids, diff).tolist()
                        # add to the end
                        para_ids_for_query += random_para_ids
                        assert len(para_ids_for_query) == (self.args.num_positive_paras + self.args.num_low_ranked_paras)
                    # now check if the first-ranked para atleast, has labels. If not, eliminate it
                    # If the first para doesnt have labels, that means none of them have
                    # because during training, the retriever makes sure that the first para has labels
                    if ex[5][q_counter][ranked_para_ids[0]].nelement() == 0:
                        continue
                    para_ids.append(para_ids_for_query)
                    filtered_q_counter.append(q_counter)

                # no para survived.
                if len(para_ids) == 0:
                    self.ret.reset()
                    return None
                ind = torch.LongTensor(para_ids).unsqueeze(2).expand(-1, -1, ex[0].size(
                        2))  # B X (num_positive_paras + num_low_ranked_paras) X max_num_words
                # first filter the documents

            filtered_q_counter_copy = copy.deepcopy(filtered_q_counter)
            filtered_q_counter_copy2 = copy.deepcopy(filtered_q_counter)
            filtered_q_counter = torch.LongTensor(filtered_q_counter).unsqueeze(1).unsqueeze(2)
            filtered_q_counter = filtered_q_counter.expand(-1, ex[0].size(1), ex[0].size(2))
            # filter the docs
            docs = torch.gather(ex[0], 0, filtered_q_counter)
            docs = torch.gather(docs, 1, ind)  # doc, this filter (gather) is to get the desired paras
            doc_masks = torch.gather(ex[2], 0, filtered_q_counter)
            doc_masks = torch.gather(doc_masks, 1, ind)  # doc_mask
            doc_feats = torch.gather(ex[1], 0, filtered_q_counter.unsqueeze(3).expand(-1, -1, -1, ex[1].size(3)))
            doc_feats = torch.gather(doc_feats, 1, ind.unsqueeze(3).expand(-1, -1, -1, len(self.args.feature_dict)))

            # also filter the spans and questions
            start = [ex[5][i] for i in filtered_q_counter_copy]
            # cloning is required because since we increase the offset of the targets later, if they are the same
            # reference, all of them increases
            start = [[start[i][j].clone() for j in para_ids[i]] for i in range(len(start))]
            end = [ex[6][i] for i in filtered_q_counter_copy]
            end = [[end[i][j].clone() for j in para_ids[i]] for i in range(len(end))]
            # # filter the question and question mask now
            filtered_q_counter_copy = torch.LongTensor(filtered_q_counter_copy).unsqueeze(1).expand(-1, ex[3].size(1))
            question = torch.gather(ex[3], 0, filtered_q_counter_copy)
            question_mask = torch.gather(ex[4], 0, filtered_q_counter_copy)
            # expand the question for all the negative samples
            question = question.unsqueeze(1).expand(-1, self.args.num_positive_paras + self.args.num_low_ranked_paras, -1)
            question_mask = question_mask.unsqueeze(1).expand(-1, self.args.num_positive_paras + self.args.num_low_ranked_paras, -1)

            # Transfer to GPU
            if torch.is_tensor(start):
                target_s = Variable(start.cuda(async=True))
                target_e = Variable(end.cuda(async=True))
            else:
                # shape of doc is B X (1 + num_low_ranked_paras) X L
                # label mask would be B X ((1 + num_low_ranked_paras) X L)
                target_s = torch.zeros(doc_masks.view(doc_masks.size(0), -1).size()).byte()
                target_e = torch.zeros(doc_masks.view(doc_masks.size(0), -1).size()).byte()
                max_num_words = docs.size(2)
                for i in range(len(start)):
                    for p in range(self.args.num_positive_paras):
                        start_index = start[i][p]
                        end_index = end[i][p]
                        if start_index.nelement() == 0:
                            continue  # happens when the number of para sent by ret is < args.num_pos_paras
                        # add offset based on the paragraph number
                        start_index += p * max_num_words
                        end_index += p * max_num_words
                        target_s[i].index_fill_(0, start_index, 1)
                        target_e[i].index_fill_(0, end_index, 1)

            inputs = [docs.view(-1, docs.size(2)), doc_feats.view(-1, doc_feats.size(2), doc_feats.size(3)),
                      doc_masks.view(-1, docs.size(2)), question.contiguous().view(-1, question.size(2)),
                      question_mask.contiguous().view(-1, question_mask.size(2))]
            if self.use_cuda:
                inputs = [e if e is None else Variable(e.cuda(async=True))
                          for e in inputs]
                target_s = Variable(target_s.cuda(async=True))
                target_e = Variable(target_e.cuda(async=True))
            else:
                inputs = [e if e is None else Variable(e) for e in inputs]
                if torch.is_tensor(start):
                    target_s = Variable(start)
                    target_e = Variable(end)
                else:
                    target_s = Variable(target_s)
                    target_e = Variable(target_e)

            all_scores_s, all_scores_e = [], []
            score_s_decode = None
            score_e_decode = None
            for r_t in range(self.args.multi_step_reading_steps):
                # 2. pass Run forward on the reader
                # 2.1 -- get reader state
                score_s, score_e, reader_state = self.network(*inputs)
                # 2.2 -- take query vector and reader state and feed it to GRU
                query = self.network.get_current_reader_query()
                query = self.multi_step_reader(query, reader_state)
                # 2.3. set the new query vector
                self.network.set_current_reader_query(query)
                # reshape score_s and score_e to B X num_paras X para-len and then to B X (num_paras * para-len)
                score_s = score_s.view_as(docs)
                score_e = score_e.view_as(docs)
                score_s = score_s.view_as(target_s)
                score_e = score_e.view_as(target_e)
                score_s_decode = F.softmax(score_s, dim=1)
                score_e_decode = F.softmax(score_e, dim=1)
                # take logsoftmax of the scores now
                score_s = F.log_softmax(score_s, dim=1)
                score_e = F.log_softmax(score_e, dim=1)
                all_scores_s.append(score_s.unsqueeze(0))
                all_scores_e.append(score_e.unsqueeze(0))
            # 2.4 reset it for next batch
            self.network.reset()
            all_scores_s = torch.cat(all_scores_s, dim=0)
            all_scores_e = torch.cat(all_scores_e, dim=0)

            if self.args.fine_tune_RL:
                # get the top most span
                top_spans = 1
                args = (score_s_decode.data.cpu(), score_e_decode.data.cpu(), top_spans, self.args.max_len)
                pred_s, pred_e, pred_score = self.decode(*args)
                # now find the actual paragraph id that the span has been found in
                # Also compute the span relative to the start of the paragraph
                num_words_in_padded_para = ex[0].size(2)
                correct_para_inds_start = [
                    [pred_s[i][j] // num_words_in_padded_para for j in range(top_spans)]
                    for i in range(len(pred_s))]
                correct_para_inds_end = [[pred_e[i][j] // num_words_in_padded_para for j in range(top_spans)]
                                         for i in range(len(pred_e))]

                # both of the above should match, otherwise it means we have an edge case
                # where the span crosses para boundaries
                for i in range(len(correct_para_inds_start)):
                    for j in range(top_spans):
                        try:
                            assert correct_para_inds_start[i][j] == correct_para_inds_end[i][j]
                        except AssertionError:
                            # As a fix, just make the start and end prediction the same.
                            pred_e[i][j] = pred_s[i][j]
                            correct_para_inds_end[i][j] = correct_para_inds_start[i][j]
                # recompute the spans relative to the beginning of the paragraph
                pred_s = [[pred_s[i][j] - correct_para_inds_start[i][j] * num_words_in_padded_para for j in
                           range(top_spans)] for i in range(len(pred_s))]
                pred_e = [[pred_e[i][j] - correct_para_inds_start[i][j] * num_words_in_padded_para for j in
                           range(top_spans)] for i in range(len(pred_e))]
                # now just keep the correct para_ids
                para_ids = [[para_ids[i][correct_para_inds_start[i][j]] for j in range(top_spans)] for i
                            in range(len(correct_para_inds_start))]
                # pred_start, pred_end, pred_scores, para_ids, ex_ids, docs
                em, f1 = self.env.get_reward(pred_s, pred_e, pred_score, para_ids, qids, ex[0], ground_truths_map)
                # reward is f1
                reward = f1
                # rearrange the scores
                logits_for_rl_loss = []
                for q_counter in range(len(sorted_para_ids_per_query)):
                    actual_idx = np.argsort(sorted_para_ids_per_query[q_counter])
                    sorted_para_scores_per_query[q_counter] = sorted_para_scores_per_query[q_counter][actual_idx]
                    # also take log-softmax
                    sorted_para_scores_per_query[q_counter] = F.log_softmax(sorted_para_scores_per_query[q_counter], dim=0)
                    logits_for_rl_loss.append(sorted_para_scores_per_query[q_counter][para_ids[q_counter]])


                # rl_loss (reward - baseline) * logp(a)
                assert len(reward) == len(logits_for_rl_loss)
                # subtract mean of reward from reward as baseline
                # mean_reward_in_batch = sum(reward)/len(reward)
                mean_reward_in_batch = 0
                reward = [r - mean_reward_in_batch for r in reward]
                rl_loss += (sum([(l * r) for (l,r) in zip(logits_for_rl_loss, reward)]))/len(reward)


            prob = torch.ones(self.args.multi_step_reading_steps).fill_(0.4)
            sum_prob = 0  # make sure there is atleast one 1
            prob_s, prob_e = None, None
            while sum_prob == 0:
                prob_s = torch.bernoulli(prob)
                sum_prob = prob_s.sum().item()
            sum_prob = 0
            while sum_prob == 0:
                prob_e = torch.bernoulli(prob)
                sum_prob = prob_e.sum().item()
            log_prob_s = Variable(torch.log(prob_s))
            log_prob_e = Variable(torch.log(prob_e))
            if self.args.cuda:
                log_prob_s = log_prob_s.cuda()
                log_prob_e = log_prob_e.cuda()

            all_scores_s += log_prob_s.unsqueeze(1).unsqueeze(2)
            all_scores_e += log_prob_e.unsqueeze(1).unsqueeze(2)

            score_s = logsumexp(all_scores_s, dim=0)
            score_e = logsumexp(all_scores_e, dim=0)

            start_logits = torch.masked_select(score_s,
                                               target_s)  # 1D tensor with number of entries equal to number of 1s in mask
            end_logits = torch.masked_select(score_e, target_e)
            # reshape back to size of target_s
            start_scores = Variable(torch.ones(target_s.size()) * float('-inf'),
                                    requires_grad=True)  # init with -inf since will be log-sum-exping next
            end_scores = Variable(torch.ones(target_e.size()) * float('-inf'), requires_grad=True)

            if self.args.cuda:
                start_scores = start_scores.cuda()
                end_scores = end_scores.cuda()
            start_scores.masked_scatter_(target_s, start_logits)
            end_scores.masked_scatter_(target_e, end_logits)

            # now logsumexp
            start_scores = logsumexp(start_scores, dim=1) - Variable(torch.log(prob_s.sum())).cuda()
            end_scores = logsumexp(end_scores, dim=1) - Variable(torch.log(prob_e.sum())).cuda()
            loss += -(start_scores.mean() + end_scores.mean()) / 2.0
            # 3. Call the multi-step-reasoner
            query_vectors = torch.index_select(query_vectors, 0, torch.LongTensor(
                filtered_q_counter_copy2).cuda())  # filter out the query vectors returned by retriever as well

            # reader state is (B * num_paras) X hidden_size
            reader_state = reader_state.view(docs.size(0), docs.size(1), -1)  # B X num_paras X hidden_size
            reader_state_mask = Variable(torch.ByteTensor(docs.size(0), docs.size(1)).fill_(0))  # all zeros
            if self.args.cuda:
                reader_state_mask = reader_state_mask.cuda()
            reader_state_wts = self.reader_self_attn(reader_state, reader_state_mask)
            reader_state = layers.weighted_avg(reader_state, reader_state_wts)

            query_vectors = self.multi_step_reasoner(query_vectors, reader_state)
            # get the closest correct para to pre-train the GRU network
            nearest_correct_para_vectors, farthest_incorrect_paras, mask = self.ret.get_nearest_correct_para_vector()

            nearest_correct_para_vectors = torch.index_select(nearest_correct_para_vectors, 0,
                                                              torch.LongTensor(filtered_q_counter_copy2).cuda())
            farthest_incorrect_paras = torch.index_select(farthest_incorrect_paras, 0,
                                                              torch.LongTensor(filtered_q_counter_copy2).cuda())
            mask = torch.index_select(mask, 0, torch.LongTensor(filtered_q_counter_copy2).cuda())
            # multi_step_loss_vec = F.mse_loss(query_vectors, nearest_correct_para_vectors, reduce=False)
            # BPE Loss
            # difference in inner product of correct and incorrect para vec

            diff = (
            torch.bmm(query_vectors.unsqueeze(1), nearest_correct_para_vectors.unsqueeze(2)).squeeze() - torch.bmm(
                query_vectors.unsqueeze(1), farthest_incorrect_paras.unsqueeze(2)).squeeze())
            #
            logits = torch.masked_select(diff, Variable(mask))
            if logits.nelement() == 0:
                flag = 1
                break
            targets = Variable(torch.cuda.FloatTensor(logits.size()).fill_(1.0)) if self.args.cuda else Variable(
                torch.FloatTensor(logits.size()).fill_(1.0))

            multi_step_loss += F.binary_cross_entropy_with_logits(logits, targets)

            # 4. Update query vectors for the retriever
            self.ret.update_query_vectors(query_vectors)


        # 5. reset for the next batch
        self.ret.reset()

        if flag == 1:  # break due to empty logit tensor
            return 0.0, ex[0].size(0)

        # Clear gradients and run backward
        self.optimizer.zero_grad()

        # beta = (1 - 0.5 * epoch_counter)
        # if beta < 0:
        #     beta = 0
        if self.args.fine_tune_RL:
            total_loss = -(rl_loss / self.args.multi_step_reasoning_steps)
        elif self.args.freeze_reader:
            total_loss = multi_step_loss/self.args.multi_step_reasoning_steps
        else:
            total_loss = (loss + multi_step_loss)/self.args.multi_step_reasoning_steps

        total_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)
        torch.nn.utils.clip_grad_norm(self.multi_step_reader.parameters(),
                                      self.args.grad_clipping)
        torch.nn.utils.clip_grad_norm(self.multi_step_reasoner.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()
        # logger.info("Multi-step-loss: {:2.4f}".format(multi_step_loss.data[0]/3.0))

        return total_loss, ex[0].size(0)

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            # Embeddings to fix are indexed after the special + N tuned words
            offset = self.args.tune_partial + self.word_dict.START
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding
            if offset < embedding.size(0):
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        """Forward a batch of examples only to get predictions.

        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()
        outputs = []
        # print("=================")
        query_norms = []
        all_query_vectors = []
        for t in range(self.args.multi_step_reasoning_steps):
            qids = ex[-1]
            # 1. get query vector, para scores and para_ids
            query_vectors, sorted_para_scores_per_query, sorted_para_ids_per_query, _ = self.dev_ret(qids)
            query_norms.append(torch.norm(query_vectors, 2, dim=1))
            all_query_vectors.append(query_vectors)
            # slice the tensors accordingly to select the top paragraph
            para_ids = []
            for ranked_para_ids in sorted_para_ids_per_query:
                if len(ranked_para_ids) <= self.args.num_paras_test:
                    # include all paragraphs
                    para_ids.append(ranked_para_ids.cpu().tolist())
                    # since gather requires the same number of elements, repeat the top para, a bit hacky
                    para_ids[-1] += [ranked_para_ids[0] for _ in range(self.args.num_paras_test - len(ranked_para_ids))]
                else:
                    para_ids.append(ranked_para_ids[:self.args.num_paras_test].tolist())

            ind = torch.LongTensor(para_ids).unsqueeze(2)  # B X num_paras_test X 1
            ind = ind.expand(-1, -1, ex[0].size(2))  # B X num_paras_test X max_num_words

            docs = torch.gather(ex[0], 1, ind).squeeze(1)  # doc
            doc_masks = torch.gather(ex[2], 1, ind).squeeze(1)  # doc_mask
            doc_feats = torch.gather(ex[1], 1,
                                     ind.unsqueeze(3).expand(-1, -1, -1, len(self.args.feature_dict))).squeeze(1)  # doc_feats

            ques = ex[3].unsqueeze(1).expand(-1, self.args.num_paras_test, -1)  # B X num_paras_test X L_q
            ques_mask = ex[4].unsqueeze(1).expand(-1, self.args.num_paras_test, -1)  # B X num_paras_test X L_q
            # reshape them as (B * num_paras_test) X para_len
            docs = docs.contiguous().view(-1, docs.size(2))
            doc_masks = doc_masks.contiguous().view(-1, doc_masks.size(2))
            doc_feats = doc_feats.contiguous().view(-1, doc_feats.size(2), doc_feats.size(3))
            ques = ques.contiguous().view(-1, ques.size(2))
            ques_mask = ques_mask.contiguous().view(-1, ques_mask.size(2))

            # Transfer to GPU
            inputs = [docs, doc_feats, doc_masks, ques, ques_mask]
            if self.use_cuda:
                inputs = [e if e is None else Variable(e.cuda(async=True))
                          for e in inputs]
            else:
                inputs = [e if e is None else Variable(e) for e in inputs]

            # 2. pass Run forward on the reader
            all_scores_s, all_scores_e = [], []
            for r_t in range(self.args.multi_step_reading_steps):
                score_s, score_e, reader_state = self.network(*inputs)
                # 2.2 -- take query vector and reader state and feed it to GRU
                query = self.network.get_current_reader_query()
                query = self.multi_step_reader(query, reader_state.view(docs.size(0), -1))
                # 2.3. set the new query vector
                self.network.set_current_reader_query(query)
                score_s = score_s.view(ex[0].size(0), self.args.num_paras_test, -1)
                score_s = F.softmax(score_s.view(ex[0].size(0), -1), dim=1)
                score_e = score_e.view(ex[0].size(0), self.args.num_paras_test, -1)
                score_e = F.softmax(score_e.view(ex[0].size(0), -1), dim=1)
                all_scores_s.append(score_s.unsqueeze(0))
                all_scores_e.append(score_e.unsqueeze(0))
            self.network.reset()
            all_scores_s = torch.cat(all_scores_s, dim=0)
            all_scores_e = torch.cat(all_scores_e, dim=0)
            # during test time avg.
            score_s = all_scores_s.mean(0)
            score_e = all_scores_e.mean(0)
            # Decode predictions
            score_s = score_s.data.cpu()
            score_e = score_e.data.cpu()
            if candidates:
                args = (score_s, score_e, candidates, top_n, self.args.max_len)
                if async_pool:
                    outputs.append([*async_pool.apply_async(self.decode_candidates, args), para_ids])
                else:
                    pred_s, pred_e, pred_score = self.decode_candidates(*args)
                    outputs.append([pred_s, pred_e, pred_score, para_ids])
            else:
                args = (score_s, score_e, self.args.top_spans, self.args.max_len)
                if async_pool:
                    outputs.append([*async_pool.apply_async(self.decode, args), para_ids])
                else:
                    pred_s, pred_e, pred_score = self.decode(*args)
                    # now find the actual paragraph id that the span has been found in
                    # Also compute the span relative to the start of the paragraph
                    num_words_in_padded_para = ex[0].size(2)
                    correct_para_inds_start = [[pred_s[i][j] // num_words_in_padded_para for j in range(self.args.top_spans)]
                                               for i in range(len(pred_s))]
                    correct_para_inds_end = [[pred_e[i][j] // num_words_in_padded_para for j in range(self.args.top_spans)]
                                               for i in range(len(pred_e))]

                    # both of the above should match, otherwise it means we have an edge case
                    # where the span crosses para boundaries
                    for i in range(len(correct_para_inds_start)):
                        for j in range(self.args.top_spans):
                            try:
                                assert correct_para_inds_start[i][j] == correct_para_inds_end[i][j]
                            except AssertionError:
                                # As a fix, just make the start and end prediction the same.
                                pred_e[i][j] = pred_s[i][j]
                                correct_para_inds_end[i][j] = correct_para_inds_start[i][j]
                    # recompute the spans relative to the beginning of the paragraph
                    pred_s = [[pred_s[i][j] - correct_para_inds_start[i][j] * num_words_in_padded_para for j in
                               range(self.args.top_spans)] for i in range(len(pred_s))]
                    pred_e = [[pred_e[i][j] - correct_para_inds_start[i][j] * num_words_in_padded_para for j in
                               range(self.args.top_spans)] for i in range(len(pred_e))]
                    # now just keep the correct para_ids
                    para_ids = [[para_ids[i][correct_para_inds_start[i][j]] for j in range(self.args.top_spans)] for i
                                in range(len(correct_para_inds_start))]
                    outputs.append([pred_s, pred_e, pred_score, para_ids])

            # 3. Call the multi-step-reasoner
            # reader state is (B * num_paras) X hidden_size
            reader_state = reader_state.view(-1, self.args.num_paras_test,
                                             self.args.doc_hidden_size)  # B X num_paras X hidden_size
            reader_state_mask = Variable(
                torch.ByteTensor(reader_state.size(0), self.args.num_paras_test).fill_(0))  # all zeros
            if self.args.cuda:
                reader_state_mask = reader_state_mask.cuda()

            reader_state_wts = self.reader_self_attn(reader_state, reader_state_mask)

            reader_state = layers.weighted_avg(reader_state, reader_state_wts)
            query_vectors = self.multi_step_reasoner(query_vectors, reader_state)
            # 4. Update query vectors for the retriever
            self.dev_ret.update_query_vectors(query_vectors)
        # 5. reset for the next batch
        self.dev_ret.reset()
        query_norms.append(torch.norm(query_vectors, 2, dim=1))
        all_query_vectors.append(query_vectors)
        return outputs, query_norms, all_query_vectors

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.

        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):

            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e. Except only consider
        spans that are in the candidates list.
        """
        pred_s = []
        pred_e = []
        pred_score = []
        for i in range(score_s.size(0)):
            # Extract original tokens stored with candidates
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            if not cands:
                # try getting from globals? (multiprocessing in pipeline mode)
                from ..pipeline.drqa import PROCESS_CANDS
                cands = PROCESS_CANDS
            if not cands:
                raise RuntimeError('No candidates given.')

            # Score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_strings=False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    # Match! Record its score.
                    scores.append(score_s[i][s] * score_e[i][e - 1])
                    s_idx.append(s)
                    e_idx.append(e - 1)

            if len(scores) == 0:
                # No candidates present
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                # Rank found candidates
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
        return pred_s, pred_e, pred_score

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        multi_step_reasoner_state_dict = copy.copy(self.multi_step_reasoner.state_dict())
        multi_step_reader_state_dict = copy.copy(self.multi_step_reader.state_dict())
        multi_step_reader_self_attn_state_dict = copy.copy(self.reader_self_attn.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'multi_step_reasoner_state_dict': multi_step_reasoner_state_dict,
            'multi_step_reader_state_dict': multi_step_reader_state_dict,
            'multi_step_reader_self_attn_state_dict': multi_step_reader_self_attn_state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
            logger.info('Model saved at {}'.format(filename))
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        state_dict = copy.copy(self.network.state_dict())
        multi_step_reasoner_state_dict = copy.copy(self.multi_step_reasoner.state_dict())
        multi_step_reader_state_dict = copy.copy(self.multi_step_reader.state_dict())
        multi_step_reader_self_attn_state_dict = copy.copy(self.reader_self_attn.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'multi_step_reasoner_state_dict': multi_step_reasoner_state_dict,
            'multi_step_reader_state_dict': multi_step_reader_state_dict,
            'multi_step_reader_self_attn_state_dict': multi_step_reader_self_attn_state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
            logger.info('Model saved at {}'.format(filename))
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        multi_step_reasoner_state_dict = None
        if 'multi_step_reasoner_state_dict' in saved_params:
            multi_step_reasoner_state_dict = saved_params['multi_step_reasoner_state_dict']
        multi_step_reader_state_dict = None
        if 'multi_step_reader_state_dict' in saved_params:
            multi_step_reader_state_dict = saved_params['multi_step_reader_state_dict']
        multi_step_reader_self_attn_state_dict = None
        if 'multi_step_reader_self_attn' in saved_params:
            multi_step_reader_self_attn_state_dict = saved_params['multi_step_reader_self_attn_state_dict']
        args = saved_params['args']
        args.word_dict = word_dict
        args.feature_dict = feature_dict
        if new_args:
            args = override_model_args(args, new_args)
        return Model(args, word_dict, feature_dict, state_dict, multi_step_reasoner_state_dict,
                     multi_step_reader_state_dict=multi_step_reader_state_dict,
                     multi_step_reader_self_attn_state_dict=multi_step_reader_self_attn_state_dict, normalize=normalize)

    @staticmethod
    def load_checkpoint(filename, new_args, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        model = Model(args, word_dict, feature_dict, state_dict=state_dict, normalize=normalize)
        model.init_optimizer(optimizer)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)


class Environment(object):

    def __init__(self, args):
        self.args = args

    def get_reward(self, pred_start, pred_end, pred_scores, para_ids, ex_ids, docs, ground_truths_map):
        """
        :param pred_start:
        :param pred_end:
        :param pred_scores:
        :param ex_ids:
        :param docs:
        :return:
        """
        em, f1 = [], []
        for i in range(len(ex_ids)):
            span_scores_map = defaultdict(float)
            max_score_i = float('-inf')
            max_span = None
            start = pred_start[i]
            end = pred_end[i]
            span_scores = pred_scores[i]
            doc_tensor = docs[i, para_ids[i]]
            for s_counter, (s, e) in enumerate(zip(start, end)):
                int_words = doc_tensor[s_counter, s:e + 1]
                predicted_span = " ".join(self.args.word_dict.ind2tok[str(w.item())] for w in int_words)
                span_scores_map[predicted_span] += span_scores[s_counter]
                if max_score_i < span_scores_map[predicted_span]:
                    max_score_i = span_scores_map[predicted_span]
                    max_span = predicted_span
            ground_truths = ground_truths_map[ex_ids[i]]
            ground_truths = list(set(ground_truths))
            em.append(utils.metric_max_over_ground_truths(utils.exact_match_score, max_span, ground_truths))
            f1.append(utils.metric_max_over_ground_truths(utils.f1_score, max_span, ground_truths))
        return em, f1
