#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
# Few methods have been adapted from https://github.com/facebookresearch/DrQA
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
import logging
from model.utils import load_embeddings
from .layers import StackedBRNN

logger = logging.getLogger()


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).
    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x = x.contiguous()

        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

class LSTMParagraphScorer(nn.Module):

    def __init__(self, args, word_dict, feature_dict):
        super(LSTMParagraphScorer, self).__init__()
        self.args = args
        self.word_dict = word_dict
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)

        if args.pretrained_words:
            self._set_embeddings()
        if args.fix_embeddings:
            for p in self.embedding.parameters():
                p.requires_grad = False

        self.document_lstm = StackedBRNN(self.args.embedding_dim, self.args.paraclf_hidden_size, 3, dropout_rate=0.2,
                                         concat_layers=True)
        self.question_lstm = StackedBRNN(self.args.embedding_dim, self.args.paraclf_hidden_size, 3, dropout_rate=0.2,
                                         concat_layers=True)
        self.para_selfaatn = LinearSeqAttn(args.paraclf_hidden_size * 6)
        self.query_selfaatn = LinearSeqAttn(args.paraclf_hidden_size * 6)
        self.bilinear = nn.Linear(args.paraclf_hidden_size * 6, args.paraclf_hidden_size * 6, bias=False)

    def _set_embeddings(self):
        # Read word embeddings.
        if not self.args.embedding_file or not self.args.pretrained_words:
            logger.warn('[ WARNING: No embeddings provided. '
                        'Keeping random initialization. ]')
            return
        logger.info('[ Loading pre-trained embeddings from {} ]'.format(self.args.embedding_file))

        embeddings = load_embeddings(self.args, self.word_dict)
        logger.info('[ Num embeddings = %d ]' % embeddings.size(0))

        # Sanity check dimensions
        new_size = embeddings.size()
        old_size = self.embedding.weight.size()

        assert (new_size[1] == old_size[1])
        if new_size[0] != old_size[0]:
            logger.warn('[ WARNING: Number of embeddings changed (%d->%d) ]' %
                        (old_size[0], new_size[0]))

        self.embedding.weight.data = embeddings

    def forward(self, x1, x1_mask, x2, x2_mask):

        """
               x1 = document word indices  [sum_paras * len_p]
               x1_mask = document padding mask [sum_paras * len_p]
               x2 = question word indices [batch * len_q]
               x2_mask = question padding mask [batch * len_q]
               """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0 and not self.training:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)

        o1 = self.document_lstm(x1_emb, x1_mask)
        o2 = self.question_lstm(x2_emb, x2_mask)

        doc_attn = self.para_selfaatn(o1, x1_mask)
        ques_attn = self.query_selfaatn(o2, x2_mask)

        doc = weighted_avg(o1, doc_attn)
        ques = weighted_avg(o2, ques_attn)

        doc = self.bilinear(doc)
        scores = ques * doc
        scores = torch.sum(scores, -1, keepdim=True)

        return scores, doc, ques