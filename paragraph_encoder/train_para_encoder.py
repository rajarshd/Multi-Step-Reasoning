import torch
import numpy as np
import json
import os
import pickle
import sys
import logging
import shutil
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler

import config
from model import utils, data, vector
from model.retriever import LSTMRetriever
from multi_corpus import MultiCorpus

from torch.utils.data.sampler import SequentialSampler, RandomSampler
import math

logger = logging.getLogger()

global_timer = utils.Timer()
stats = {'timer': global_timer, 'epoch': 0, 'best_valid': 0, 'best_verified_valid': 0, 'best_acc': 0, 'best_verified_acc': 0}

def make_data_loader(args, corpus, train_time=False):

    dataset = data.MultiCorpusDataset(
        args,
        corpus,
        args.word_dict,
        args.feature_dict,
        single_answer=False,
        para_mode=args.para_mode,
        train_time=train_time
    )
    sampler = SequentialSampler(dataset) if not train_time else RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify(args, args.para_mode, train_time=train_time),
        pin_memory=True
    )

    return loader



def init_from_checkpoint(args):

    logger.info('Loading model from saved checkpoint {}'.format(args.pretrained))
    model = torch.load(args.pretrained)
    word_dict = model['word_dict']
    feature_dict = model['feature_dict']

    args.vocab_size = len(word_dict)
    args.embedding_dim_orig = args.embedding_dim
    args.word_dict = word_dict
    args.feature_dict = feature_dict

    ret = LSTMRetriever(args, word_dict, feature_dict)
    # load saved param values
    ret.model.load_state_dict(model['state_dict']['para_clf'])
    optimizer = None
    parameters = ret.get_trainable_params()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(parameters,
                                 weight_decay=args.weight_decay)
    elif args.optimizer == 'nag':
        optimizer = NAG(parameters, args.learning_rate, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        raise RuntimeError('Unsupported optimizer: %s' % args.optimizer)
    optimizer.load_state_dict(model['state_dict']['optimizer'])
    logger.info('Model loaded...')
    return ret, optimizer, word_dict, feature_dict


def init_from_scratch(args, train_exs):

    logger.info('Initializing model from scratch')
    word_dict = feature_dict = None
    # create or get vocab
    word_dict = utils.build_word_dict(args, train_exs)
    if word_dict is not None:
        args.vocab_size = len(word_dict)
    args.embedding_dim_orig = args.embedding_dim
    args.word_dict = word_dict
    args.feature_dict = feature_dict

    ret = LSTMRetriever(args, word_dict, feature_dict)

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    # --------------------------------------------------------------------------
    # train
    parameters = ret.get_trainable_params()


    optimizer = None
    if parameters is not None and len(parameters) > 0:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(parameters, args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optimizer == 'adamax':
            optimizer = optim.Adamax(parameters,
                                     weight_decay=args.weight_decay)
        elif args.optimizer == 'nag':
            optimizer = NAG(parameters, args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % args.optimizer)
    else:
        pass

    return ret, optimizer, word_dict, feature_dict

def train_binary_classification(args, ret_model, optimizer, train_loader, verified_dev_loader=None):

    args.train_time = True
    para_loss = utils.AverageMeter()
    ret_model.model.train()
    for idx, ex in enumerate(train_loader):
        if ex is None:
            continue

        inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda(async=True))
                  for e in ex[:]]
        ret_input = [*inputs[:4]]
        scores, _, _ = ret_model.score_paras(*ret_input)
        y_num_occurrences = Variable(ex[-2])
        labels = (y_num_occurrences > 0).float()
        labels = labels.cuda()
        # BCE logits loss
        batch_para_loss = F.binary_cross_entropy_with_logits(scores.squeeze(1), labels)
        optimizer.zero_grad()
        batch_para_loss.backward()

        torch.nn.utils.clip_grad_norm(ret_model.get_trainable_params(),
                                      2.0)
        optimizer.step()
        para_loss.update(batch_para_loss.data.item())
        if math.isnan(para_loss.avg):
            import pdb
            pdb.set_trace()

        if idx % 25 == 0 and idx > 0:
            logger.info('Epoch = {} | iter={}/{} | para loss = {:2.4f}'.format(
                stats['epoch'],
                idx, len(train_loader),
                para_loss.avg))
            para_loss.reset()


def eval_binary_classification(args, ret_model, corpus, dev_loader, verified_dev_loader=None, save_scores = True):
    total_exs = 0
    args.train_time = False
    ret_model.model.eval()
    accuracy = 0.0
    for idx, ex in enumerate(tqdm(dev_loader)):
        if ex is None:
            raise BrokenPipeError

        inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda(async=True))
                  for e in ex[:]]
        ret_input = [*inputs[:4]]
        total_exs += ex[0].size(0)

        scores, _, _ = ret_model.score_paras(*ret_input)

        scores = F.sigmoid(scores)
        y_num_occurrences = Variable(ex[-2])
        labels = (y_num_occurrences > 0).float()
        labels = labels.data.numpy()
        scores = scores.cpu().data.numpy()
        scores = scores.reshape((-1))
        if save_scores:
            for i, pid in enumerate(ex[-1]):
                corpus.paragraphs[pid].model_score = scores[i]

        scores = scores > 0.5
        a = scores == labels
        accuracy += a.sum()

    logger.info('Eval accuracy = {} '.format(accuracy/total_exs))
    top1 = get_topk(corpus)
    return top1

def print_vectors(args, para_vectors, question_vectors, corpus, train=False, test=False):
    all_question_vectors = []
    all_para_vectors = []
    qid2idx = {}
    cum_num_lens = []
    all_correct_ans = {}
    cum_num_len = 0
    for question_i, qid in enumerate(corpus.questions):
        labels = []
        all_question_vectors.append(question_vectors[qid])
        qid2idx[qid] = question_i
        cum_num_len += len(corpus.questions[qid].pids)
        cum_num_lens.append(cum_num_len)
        for para_i, pid in enumerate(corpus.questions[qid].pids):
            if corpus.paragraphs[pid].ans_occurance > 0:
                labels.append(para_i)
            all_para_vectors.append(para_vectors[pid])
        all_correct_ans[qid] = labels
    all_para_vectors = np.stack(all_para_vectors)
    all_question_vectors = np.stack(all_question_vectors)
    assert all_para_vectors.shape[0] == cum_num_lens[-1]
    assert all_question_vectors.shape[0] == len(cum_num_lens)
    assert all_question_vectors.shape[0] == len(qid2idx)
    assert all_question_vectors.shape[0] == len(all_correct_ans)

    ## saving code
    if train:
        OUT_DIR = os.path.join(args.save_dir, args.src,  args.domain, "train/")
    else:
        if args.is_test == 0:
            OUT_DIR = os.path.join(args.save_dir, args.src, args.domain, "dev/")
        else:
            OUT_DIR = os.path.join(args.save_dir, args.src, args.domain, "test/")


    logger.info("Printing vectors at {}".format(OUT_DIR))
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    else:
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        os.makedirs(OUT_DIR)


    json.dump(qid2idx, open(OUT_DIR + 'map.json', 'w'))
    json.dump(all_correct_ans, open(OUT_DIR + 'correct_paras.json', 'w'))
    all_cumlen = np.array(cum_num_lens)
    np.save(OUT_DIR + "document", all_para_vectors)
    np.save(OUT_DIR + "question", all_question_vectors)
    np.save(OUT_DIR + "all_cumlen", cum_num_lens)


def save_vectors(args, ret_model, corpus, data_loader, verified_dev_loader=None, save_scores = True, train=False, test=False):
    total_exs = 0
    args.train_time = False
    ret_model.model.eval()
    para_vectors = {}
    question_vectors = {}
    for idx, ex in enumerate(tqdm(data_loader)):
        if ex is None:
            raise BrokenPipeError

        inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda(async=True))
                  for e in ex[:]]
        ret_input = [*inputs[:4]]
        total_exs += ex[0].size(0)

        scores, doc, ques = ret_model.score_paras(*ret_input)
        scores = scores.cpu().data.numpy()
        scores = scores.reshape((-1))

        if save_scores:
            for i, pid in enumerate(ex[-1]):
                para_vectors[pid] = doc[i]
            for i, qid in enumerate([corpus.paragraphs[pid].qid for pid in ex[-1]]):
                if qid not in question_vectors:
                    question_vectors[qid] = ques[i]
            for i, pid in enumerate(ex[-1]):
                corpus.paragraphs[pid].model_score = scores[i]

    get_topk(corpus)
    print_vectors(args, para_vectors, question_vectors, corpus, train, test)



def get_topk(corpus):
    top1 = 0
    top3 = 0
    top5 = 0
    for qid in corpus.questions:

        para_scores = [(corpus.paragraphs[pid].model_score,corpus.paragraphs[pid].ans_occurance ) for pid in corpus.questions[qid].pids]
        sorted_para_scores = sorted(para_scores, key=lambda x: x[0], reverse=True)

        if sorted_para_scores[0][1] > 0:
            top1 += 1
        if sum([ans[1] for ans in sorted_para_scores[:3]]) > 0:
            top3 += 1
        if sum([ans[1] for ans in sorted_para_scores[:5]]) > 0:
            top5 += 1

    top1 = top1/len(corpus.questions)
    top3 = top3/len(corpus.questions)
    top5 = top5/len(corpus.questions)

    logger.info('top1 = {}, top3 = {}, top5 = {} '.format(top1, top3 ,top5 ))
    return top1

def get_topk_tfidf(corpus):
    top1 = 0
    top3 = 0
    top5 = 0
    for qid in corpus.questions:

        para_scores = [(corpus.paragraphs[pid].tfidf_score, corpus.paragraphs[pid].ans_occurance) for pid in
                       corpus.questions[qid].pids]
        sorted_para_scores = sorted(para_scores, key=lambda x: x[0])
        # import pdb
        # pdb.set_trace()
        if sorted_para_scores[0][1] > 0:
            top1 += 1
        if sum([ans[1] for ans in sorted_para_scores[:3]]) > 0:
            top3 += 1
        if sum([ans[1] for ans in sorted_para_scores[:5]]) > 0:
            top5 += 1

    logger.info(
        'top1 = {}, top3 = {}, top5 = {} '.format(top1 / len(corpus.questions), top3 / len(corpus.questions),
                                                  top5 / len(corpus.questions)))


def run_predictions(args, data_loader, model, eval_on_train_set=False):

    args.train_time = False
    top_1 = 0
    top_3 = 0
    top_5 = 0
    total_num_questions = 0
    map_counter = 0
    cum_num_lens = []
    qid2idx = {}
    sum_num_paras = 0
    all_correct_answers = {}

    for ex_counter, ex in tqdm(enumerate(data_loader)):

        ret_input = [*ex]
        y_num_occurrences = ex[3]
        labels = (y_num_occurrences > 0)
        try:
            topk_paras, docs, ques = model.return_topk(5,*ret_input)
        except RuntimeError:
            import pdb
            pdb.set_trace()

        num_paras = ex[1]
        qids = ex[-1]

        if args.save_para_clf_output:
            docs = docs.cpu().data.numpy()
            ques = ques.cpu().data.numpy()
            if ex_counter == 0:
                documents = docs
                questions = ques
            else:
                documents = np.concatenate([documents, docs])
                questions = np.concatenate([questions, ques])


            ### create map and cum_num_lens

            for i, qid in enumerate(qids):
                qid2idx[qid] = map_counter
                sum_num_paras += num_paras[i]
                cum_num_lens.append(sum_num_paras)
                all_correct_answers[map_counter] = []

                st = sum(num_paras[:i])
                for j in range(num_paras[i]):
                    if labels[st+j] == 1:
                        all_correct_answers[map_counter].append(j)

                ### Test case:
                assert len(all_correct_answers[map_counter]) == sum(labels.data.numpy()[st: st + num_paras[i]])

                map_counter += 1



        counter = 0
        for q_counter, ranked_para_ids in enumerate(topk_paras):
            total_num_questions += 1
            for i, no_paras in enumerate(ranked_para_ids):
                if labels[counter + no_paras ] ==1:
                    if i <= 4:
                        top_5 += 1
                    if i <= 2:
                        top_3 += 1
                    if i <= 0:
                        top_1 += 1
                    break
            counter += num_paras[q_counter]



    logger.info('Accuracy of para classifier when evaluated on the annotated dev set.')
    logger.info('top-1: {:2.4f}, top-3: {:2.4f}, top-5: {:2.4f}'.format(
        (top_1 * 1.0 / total_num_questions),
        (top_3 * 1.0 / total_num_questions),
        (top_5 * 1.0 / total_num_questions)))


    ## saving code
    if args.save_para_clf_output:
        if eval_on_train_set:
            OUT_DIR = "/iesl/canvas/sdhuliawala/vectors_web/train/"
        else:
            OUT_DIR = "/iesl/canvas/sdhuliawala/vectors_web/dev/"

        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)
        else:
            shutil.rmtree(OUT_DIR, ignore_errors=True)
            os.mkdir(OUT_DIR)


        #Test cases
        assert cum_num_lens[-1] == documents.shape[0]
        assert questions.shape[0] == documents.shape[0]
        assert len(cum_num_lens) == len(qid2idx)
        assert len(cum_num_lens) == len(all_correct_answers)

        json.dump(qid2idx, open(OUT_DIR + 'map.json', 'w'))
        json.dump(all_correct_answers, open(OUT_DIR + 'correct_paras.json', 'w'))
        all_cumlen = np.array(cum_num_lens)
        np.save(OUT_DIR + "document", documents)
        np.save(OUT_DIR + "question", questions)
        np.save(OUT_DIR + "all_cumlen", all_cumlen)
    return (top_1 * 1.0 / total_num_questions), (top_3 * 1.0 / total_num_questions), (top_5 * 1.0 / total_num_questions)


def save(args, model, optimizer, filename, epoch=None):

    params = {
        'state_dict': {
            'para_clf': model.state_dict(),
            'optimizer': optimizer.state_dict()
        },
        'word_dict': args.word_dict,
        'feature_dict': args.feature_dict
    }
    args.word_dict = None
    args.feature_dict = None
    params['config'] = vars(args)
    if epoch:
        params['epoch'] = epoch
    try:
        torch.save(params, filename)
        # bad hack for not saving dictionary twice
        args.word_dict = params['word_dict']
        args.feature_dict = params['feature_dict']
    except BaseException:
        logger.warn('[ WARN: Saving failed... continuing anyway. ]')


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------

def main(args):

    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % json.dumps(vars(args), indent=4, sort_keys=True))

    # small can't test
    if args.small == 1:
        args.test = 0

    if args.small == 1:
        args.train_file_name = args.train_file_name + "_small"
        args.dev_file_name = args.dev_file_name + "_small"
        if args.test == 1:
            args.test_file_name = args.test_file_name + "_small"

    args.train_file_name = args.train_file_name + ".pkl"
    args.dev_file_name = args.dev_file_name + ".pkl"
    if args.test == 1:
        args.test_file_name = args.test_file_name + ".pkl"

    logger.info("Loading pickle files")
    fin = open(os.path.join(args.data_dir, args.src, "data", args.domain, args.train_file_name), "rb")
    all_train_exs = pickle.load(fin)
    fin.close()

    fin = open(os.path.join(args.data_dir,  args.src, "data", args.domain, args.dev_file_name), "rb")
    all_dev_exs = pickle.load(fin)
    fin.close()
    if args.test == 1:
        fin = open(os.path.join(args.data_dir,  args.src, "data", args.domain, args.test_file_name), "rb")
        all_test_exs = pickle.load(fin)
        fin.close()

    logger.info("Loading done!")

    logger.info("Num train examples {}".format(len(all_train_exs.paragraphs)))
    logger.info("Num dev examples {}".format(len(all_dev_exs.paragraphs)))
    if args.test == 1:
        logger.info("Num test examples {}".format(len(all_test_exs.paragraphs)))


    if args.pretrained is None:
        ret_model, optimizer, word_dict, feature_dict = init_from_scratch(args, all_train_exs)
    else:
        ret_model, optimizer, word_dict, feature_dict = init_from_checkpoint(args)

    # make data loader
    logger.info("Making data loaders...")
    if word_dict == None:
        args.word_dict = utils.build_word_dict(args, (all_train_exs, all_dev_exs))
        word_dict = args.word_dict

    train_loader = make_data_loader(args, all_train_exs, train_time=False) if args.eval_only else make_data_loader(args, all_train_exs, train_time=True)
    dev_loader = make_data_loader(args, all_dev_exs)
    if args.test:
        test_loader = make_data_loader(args, all_test_exs)


    if args.eval_only:
        logger.info("Saving dev paragraph vectors")
        save_vectors(args, ret_model, all_dev_exs, dev_loader, verified_dev_loader=None)


        logger.info("Saving train paragraph vectors")
        save_vectors(args, ret_model, all_train_exs, train_loader, verified_dev_loader=None, train=True)
        if args.test:
            args.is_test = 1
            logger.info("Saving test paragraph vectors")
            save_vectors(args, ret_model, all_test_exs, test_loader, verified_dev_loader=None)

    else:
        get_topk_tfidf(all_dev_exs)
        for epoch in range(args.num_epochs):
            stats['epoch'] = epoch
            train_binary_classification(args, ret_model, optimizer, train_loader, verified_dev_loader=None)
            logger.info('checkpointing  model at {}'.format(args.model_file))
            ## check pointing##
            save(args, ret_model.model, optimizer, args.model_file+".ckpt", epoch=stats['epoch'])

            logger.info("Evaluating on the full dev set....")
            top1 = eval_binary_classification(args, ret_model, all_dev_exs, dev_loader, verified_dev_loader=None)
            if stats['best_acc'] < top1:
                stats['best_acc'] = top1
                logger.info('Best accuracy {}'.format(stats['best_acc']))
                logger.info('Saving model at {}'.format(args.model_file))
                logger.info("Logs saved at {}".format(args.log_file))
                save(args, ret_model.model, optimizer, args.model_file, epoch=stats['epoch'])


if __name__ == '__main__':
    # MODEL
    logger.info('-' * 100)
    # Parse cmdline args and setup environment
    args = config.get_args()

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
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
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
    logger.info('[ COMMAND: %s ]' % ' '.join(sys.argv))

    # Run!
    main(args)