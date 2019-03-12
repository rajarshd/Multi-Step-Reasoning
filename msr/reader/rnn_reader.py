#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers
import logging

logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class StackedRNNCell(nn.Module):
    """
    impl of stacked rnn cell.
    """

    def __init__(self, args, cell_type, in_size, h_size, num_layers=3):
        super(StackedRNNCell, self).__init__()
        self.cells = nn.ModuleList()
        self.args = args
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.rnn = None
        if cell_type == 'lstm':
            self.rnn = nn.LSTMCell
        elif cell_type == 'gru':
            self.rnn = nn.GRUCell
        elif cell_type == 'rnn':
            self.rnn = nn.RNNCell
        if self.rnn is None:
            logger.info('[ Defaulting to LSTM cell_type ]')
            self.cell_type = 'lstm'
            self.rnn = nn.LSTMCell

        for _ in range(num_layers):
            if args.cuda:
                    self.cells.append(self.rnn(in_size, h_size).cuda())
            else:
                self.cells.append(self.rnn(in_size, h_size))
            in_size = h_size

    def forward(self, x, hiddens):

        """
        :param x: input embedding
        :param hiddens: an array of length num_layers, hiddens[j] contains (h_t, c_t) of jth layer
        :return:
        """
        input = x
        hiddens_out = []
        for l in range(self.num_layers):
            h_out = self.cells[l](input, hiddens[l])
            hiddens_out.append(h_out)
            input = h_out[0] if self.cell_type == 'lstm' else h_out
        return hiddens_out


class MultiStepReasoner(nn.Module):
    """
    does multistep reasoning by taking the reader state and the previous query to generate a new query
    """

    def __init__(self, args, input_dim, hidden_dim):
        super(MultiStepReasoner, self).__init__()
        self.args = args
        self.gru_cell = StackedRNNCell(args, 'gru', input_dim, hidden_dim, self.args.num_gru_layers)
        self.args.cuda = True
        self.linear1 = nn.Linear(self.args.num_gru_layers * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        if self.args.cuda:
            self.gru_cell = self.gru_cell.cuda()
            self.linear1 = self.linear1.cuda()
            self.linear2 = self.linear2.cuda()

    def forward(self, retriever_query, reader_state):
        hiddens = self.gru_cell(reader_state, [retriever_query for _ in range(self.args.num_gru_layers)])
        hiddens = torch.cat(hiddens, dim=1)
        # pass it through a MLP
        out = self.linear2(F.relu(self.linear1(hiddens)))
        return out


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        self.question_hidden, self.doc_hiddens = None, None

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        self.args.doc_hidden_size = doc_hidden_size
        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        #  this is for computing the reader state
        self.reader_state_self_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        if self.question_hidden is None:  # read the paras only once
            # Embed both document and question
            x1_emb = self.embedding(x1)
            x2_emb = self.embedding(x2)

            # Dropout on embeddings
            if self.args.dropout_emb > 0:
                x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                               training=self.training)
                x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                               training=self.training)

            # Form document encoding inputs
            drnn_input = [x1_emb]
            # import pdb
            # pdb.set_trace()
            # Add attention-weighted question representation
            if self.args.use_qemb and self.question_hidden is None:
                x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
                drnn_input.append(x2_weighted_emb)

            # Add manual features
            if self.args.num_features > 0:
                drnn_input.append(x1_f)

            # Encode document with RNN
            self.doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)


            # Encode question with RNN + merge hiddens
            if self.question_hidden is None:
                question_hiddens = self.question_rnn(x2_emb, x2_mask)
                if self.args.question_merge == 'avg':
                    q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
                elif self.args.question_merge == 'self_attn':
                    q_merge_weights = self.self_attn(question_hiddens, x2_mask)
                self.question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # reader_state = torch.max(doc_hiddens, dim=1)[0]
        reader_state_weights = F.softmax(self.reader_state_self_attn(self.doc_hiddens, self.question_hidden, x1_mask), dim=1)
        reader_state = layers.weighted_avg(self.doc_hiddens, reader_state_weights)

        # Predict start and end positions
        start_scores = self.start_attn(self.doc_hiddens, self.question_hidden, x1_mask)
        end_scores = self.end_attn(self.doc_hiddens, self.question_hidden, x1_mask)
        return start_scores, end_scores, reader_state


    def reset(self):
        self.question_hidden = None
        self.doc_hiddens = None

    def get_current_reader_query(self):
        return self.question_hidden

    def set_current_reader_query(self, query):
        self.question_hidden = query