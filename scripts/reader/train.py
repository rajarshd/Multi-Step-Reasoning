#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main DrQA reader training script."""

import socket
import argparse
import torch
import numpy as np
import json
import os
import sys
import subprocess
import logging
from tqdm import tqdm
import pickle
from collections import defaultdict


from msr.reader import utils, vector, config, data
from msr.reader.model import Model

from paragraph_encoder.multi_corpus import MultiCorpus

from torch.utils.data.sampler import SequentialSampler, RandomSampler
logger = logging.getLogger()

# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults

ROOT_DIR = '.'

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining tos training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=10,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda'
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=64,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=2,
                         help='Batch size during validation/testing')
    runtime.add_argument('--multi_step_reasoning_steps', type=int, default=3,
                         help='Number of steps of mult-step-reasoning')
    runtime.add_argument('--multi_step_reading_steps', type=int, default=1,
                         help='Number of steps of mult-step-reading')
    runtime.add_argument('--dropout-san-prediction', type=float, default=0.4,
                       help='During training, dropout few predictions')
    runtime.add_argument('--num_gru_layers', type=int, default=3,
                         help='Number of layers of GRU')
    runtime.add_argument('--domain', type=str, default="web-open",
                         help='wiki/web/web-open')
    runtime.add_argument('--dataset_name', type=str, default="triviaqa",
                         help='triviaqa/searchqa/')
    runtime.add_argument('--freeze_reader', type=int, default=0,
                         help='Donot train the reader?')
    runtime.add_argument('--fine_tune_RL', type=int, default=0,
                         help='Keep everything fixed, fine tune reasoner with RL')
    runtime.add_argument('--test', type=int, default=0,
                         help='eval on test data?')
    runtime.add_argument('--drqa_plus', type=int, default=1,
                         help='Use reader of DrQA++')
    runtime.add_argument('--num_positive_paras', type=int, default=1,
                         help='DrQA++ relies on few paras containing the answer, '
                              'returned by the retriever during training. Default 1')
    runtime.add_argument('--num_paras_test', type=int, default=15,
                         help='Number of paras to read at test time. Default 1')
    runtime.add_argument('--num_low_ranked_paras', type=int, default=0,
                         help='DrQA++ relies on few low ranked paras by the retriever during training.')
    runtime.add_argument('--cheat', type=int, default=0,
                         help='at test time, overwrite the retriever output and put correct paras containign annotations')
    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model_dir', type=str, default="",
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       default='SQuAD-v1.1-train-processed-corenlp.txt',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='SQuAD-v1.1-dev-processed-corenlp.txt',
                       help='Preprocessed dev file')
    files.add_argument('--dev-json', type=str, default='SQuAD-v1.1-dev.json',
                       help=('Unprocessed dev file to run validation '
                             'while training on'))
    files.add_argument('--embed-dir', type=str, default="",
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='crawl-300d-2M.txt',
                       help='Space-separated pretrained embeddings file')
    files.add_argument('--official_output_json', type=str, default="official_output.json",
                       help='Directory of pre-trained embedding files')
    files.add_argument('--saved_para_vectors_dir', type=str,
                       help='Directory where para and query vectors are saved by the retrievers')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=True,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    save_load.add_argument('--create_vocab', type=int, default=0,
                           help='Create vocab or load saved')
    save_load.add_argument('--vocab_dir', type=str, default="")
    save_load.add_argument('--embedding_table_path', type=str, default='embedding_table.mdl')
    save_load.add_argument('--save_pickle_files', type=int, default=0,
                           help='Save the processed train, dev files for faster loading')
    save_load.add_argument('--load_pickle_files', type=int, default=1,
                           help='Load the processed train, dev files for faster loading')
    save_load.add_argument('--small', type=int, default=0,
                           help='Experiment on small files (for debugging)')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')
    preprocess.add_argument('--use_pretrained_para_clf', type=int, default=1, help=" use pretrained para clf...")
    preprocess.add_argument('--require_answer', type=int, default=0,
                            help="Retriever only sends paragraphs which have the answers")

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--eval_only', type=int, default=0,
                         help='Evaluate only after loading a pretrained model')
    general.add_argument('--valid-metric', type=str, default='f1',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')


def make_data_loader(args, exs, train_time=False):

    dataset = data.ReaderDataset(
        args,
        exs,
        args.word_dict,
        args.feature_dict,
        single_answer=False,
        train_time=train_time
    )
    sampler = SequentialSampler(dataset) if not train_time else RandomSampler(dataset)
    batch_size = args.batch_size if train_time else args.test_batch_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=vector.batchify,
        pin_memory=True
    )

    return loader


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Set model name
    args.vocab_dir = os.path.join(args.data_dir, args.dataset_name, "vocab", args.domain)
    args.embedding_file = os.path.join(args.data_dir, args.dataset_name, "embeddings", args.embedding_file)
    args.embedding_table_path = os.path.join(args.data_dir, args.dataset_name, "embeddings", args.domain,
                                             args.embedding_table_path)
    args.origin_data_dir = args.data_dir
    args.data_dir = os.path.join(args.data_dir, args.dataset_name, "data", args.domain)
    if os.path.exists(args.embedding_table_path):
        args.embedding_table = True
    else:
        args.embedding_table = False
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    if args.small == 0:  # only save on full experiments, saves disk space
        args.model_dir = os.path.join(args.model_dir, args.dataset_name, "expts", args.model_name)
        subprocess.call(['mkdir', '-p', args.model_dir])
        # subprocess.call(['cp', '-r', ROOT_DIR, args.model_dir])
        # Set log + model file names
        args.log_file = os.path.join(args.model_dir, 'log.txt')
        args.model_file = os.path.join(args.model_dir, 'model.mdl')
    else:
        args.model_file = ""
        args.model_dir = ""
        args.log_file = None

    args.official_output_json = os.path.join(args.model_dir, args.official_output_json)
    args.use_pretrained_para_clf = (args.use_pretrained_para_clf == 1)
    args.create_vocab = (args.create_vocab == 1)

    args.eval_only = (args.eval_only == 1)
    args.require_answer = (args.require_answer == 1)
    args.drqa_plus = (args.drqa_plus == 1)
    # args.saved_para_vectors_dir = os.path.join(DATA_DIR, args.dataset_name, 'paragraph_vectors', args.domain)
    args.freeze_reader = (args.freeze_reader == 1)
    args.cheat = (args.cheat == 1)
    args.fine_tune_RL = (args.fine_tune_RL == 1)
    if args.fine_tune_RL:
        assert args.freeze_reader is True

    args.test = (args.test == 1)

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args, train_exs)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, train_exs, dev_exs)
    logger.info('Num words = %d' % len(word_dict))
    # Initialize model
    logger.info('Initializing model')
    model = Model(args, word_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(args, word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats, ground_truths_map):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):

        ret = model.update(ex, epoch_counter=global_stats['epoch'], ground_truths_map=ground_truths_map)
        if ret is None:
            continue
        train_loss.update(*ret)
        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint and (args.small == 0):
        logger.info("Checkpointing...")
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)

def validate_official(args, data_loader, model, global_stats,
                      offsets, texts, answers, ground_truths_map=None, official_eval_output=False):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.

    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Run through examples
    examples = 0
    official_output_json = {}
    fout = None
    if args.eval_only:
        fout = open(os.path.join(args.model_dir, "outputs.txt"), "w")
    for ex in tqdm(data_loader):
        ex_id, batch_size = ex[-1], ex[0].size(0)
        outputs, query_norms, all_query_vectors = model.predict(ex)
        max_scores, max_spans = [], []
        for i in range(ex[0].size(0)):
            span_scores_map = defaultdict(float)
            max_score_i = float('-inf')
            max_span = None
            for step_counter, output in enumerate(outputs):   # for each time step
                pred_s, pred_e, pred_score, para_ids = output
                start = pred_s[i]
                end = pred_e[i]
                span_scores = pred_score[i]
                doc_tensor = ex[0][i, para_ids[i]]
                for s_counter, (s, e) in enumerate(zip(start, end)):
                    int_words = doc_tensor[s_counter, s:e+1]
                    predicted_span = " ".join(args.word_dict.ind2tok[str(w.item())] for w in int_words)
                    span_scores_map[predicted_span] += span_scores[s_counter]
                    if max_score_i < span_scores_map[predicted_span]:
                        max_score_i = span_scores_map[predicted_span]
                        max_span = predicted_span

            max_scores.append(max_score_i)
            max_spans.append(max_span)
            # calculate em and f1
            ground_truths = ground_truths_map[ex_id[i]]
            ground_truths = list(set(ground_truths))
            em = utils.metric_max_over_ground_truths(utils.exact_match_score, max_span, ground_truths)
            exact_match.update(em)
            f1.update(utils.metric_max_over_ground_truths(utils.f1_score, max_span, ground_truths))
            examples += 1
            official_output_json[ex_id[i]] = max_span
    if fout is not None:
        fout.close()
    logger.info('dev valid official: Epoch = %d | EM = %.2f | ' %
                (global_stats['epoch'], exact_match.avg * 100) +
                'F1 = %.2f | examples = %d | valid time = %.2f (s)' %
                    (f1.avg * 100, examples, eval_time.time()))

    logger.info("Writing official output at {}".format(args.official_output_json))
    json.dump(official_output_json, open(args.official_output_json, "w"))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}


def eval_accuracies(pred_s, target_s, pred_e, target_e):
    """An unofficial evalutation helper.
    Compute exact start/end/complete match accuracies for a batch.
    """
    # Convert 1D tensors to lists of lists (compatibility)
    if torch.is_tensor(target_s):
        target_s = [[e] for e in target_s]
        target_e = [[e] for e in target_e]

    # Compute accuracies from targets
    batch_size = len(pred_s)
    start = utils.AverageMeter()
    end = utils.AverageMeter()
    em = utils.AverageMeter()
    for i in range(batch_size):
        # Start matches
        if pred_s[i] in target_s[i]:
            start.update(1)
        else:
            start.update(0)

        # End matches
        if pred_e[i] in target_e[i]:
            end.update(1)
        else:
            end.update(0)

        # Both start and end match
        if any([1 for _s, _e in zip(target_s[i], target_e[i])
                if _s == pred_s[i] and _e == pred_e[i]]):
            em.update(1)
        else:
            em.update(0)
    return start.avg * 100, end.avg * 100, em.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    max_para_len = 400
    logger.info("Domain: {}".format(args.domain))

    train_exs, dev_exs = None, None
    if args.small == 1:
        train_file_name = "processed_train_small.pkl"
        dev_file_name = "processed_dev_small.pkl"
    else:
        train_file_name = "processed_train.pkl"
        dev_file_name = "processed_test.pkl" if args.test else "processed_dev.pkl"

    logger.info("Loading pickle files")
    fin = open(os.path.join(args.data_dir, train_file_name), "rb")
    train_exs = pickle.load(fin)
    fin.close()
    fin = open(os.path.join(args.data_dir, dev_file_name), "rb")
    dev_exs = pickle.load(fin)
    fin.close()
    logger.info("Loading done!")

    logger.info('Num train examples = %d' % len(train_exs.questions))
    # dev_exs = utils.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs.questions))


    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = Model.load_checkpoint(checkpoint_file, args)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = Model.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = utils.load_words(args, train_exs + dev_exs)
                added = model.expand_dictionary(words)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)
        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_exs, dev_exs)

        # Set up partial tuning of embeddings
        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' %
                        args.tune_partial)
            top_words = utils.top_question_words(
                args, train_exs, model.word_dict
            )
            for word in top_words[:5]:
                logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]:
                logger.info(word)
            model.tune_embeddings([w[0] for w in top_words])

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info('-' * 100)
    logger.info('Make data loaders')

    args.word_dict = model.word_dict
    args.feature_dict = model.feature_dict

    # train_dataset = data.ReaderDataset(train_exs, model, single_answer=True)

    train_loader = make_data_loader(args, train_exs, train_time=True)
    dev_loader = make_data_loader(args, dev_exs)

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    logger.info('-' * 100)
    logger.info("Reading ground truths for train")
    fin = open(os.path.join(args.data_dir, "train_testing.txt"))
    train_ground_truths_map = {}
    for line in fin:
        line = line.strip()
        qid, ground_truth = line.split("\t")
        train_ground_truths_map[qid] = ground_truth.split(
            "$@#$@#")  # this is the special char with which the gt ans are separated
    fin.close()
    logger.info("Reading ground truths for dev")
    fin = open(os.path.join(args.data_dir, "test_testing.txt")) if args.test else open(
        os.path.join(args.data_dir, "dev_testing.txt"))
    dev_ground_truths_map = {}
    for line in fin:
        line = line.strip()
        qid, ground_truth = line.split("\t")
        dev_ground_truths_map[qid] = ground_truth.split(
            "$@#$@#")  # this is the special char with which the gt ans are separated
    fin.close()

    if args.eval_only:
        logger.info("Eval only mode")
        result = validate_official(args, dev_loader, model, stats, None, None, None,
                                   ground_truths_map=dev_ground_truths_map, official_eval_output=True)
        logger.info("Exiting...")
        sys.exit(0)

    logger.info('Starting training...')
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch
        # Train
        train(args, train_loader, model, stats, train_ground_truths_map)
        # Validate official
        if args.official_eval:
            result = validate_official(args, dev_loader, model, stats, None, None, None, ground_truths_map=dev_ground_truths_map)

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
