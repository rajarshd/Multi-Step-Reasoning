#!/usr/bin/env python3
import argparse
import os
from smart_open import smart_open
import subprocess
import logging
import uuid
import time


logger = logging.getLogger()

USER = os.getenv('USER')

SRC_DIR = "."

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    parser.add_argument('--no_cuda', type='bool', default=False)
    parser.add_argument('--test_time_cuda', type='bool', default=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--data_workers', type=int, default=5)
    # Basics
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint', type='bool', default=True)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--uncased_question', type='bool', default=False)
    parser.add_argument('--uncased_doc', type='bool', default=False)
    parser.add_argument('--use_only_distant_sup', type=int, default=1,
                        help="For hotpotQA use only string matching as supervision or the relevant paragraph supervision given by them")

    # vocab dir
    parser.add_argument('--create_vocab', type=int, default=0,
                        help='Create vocab files and write them even if they exist')

    # Data files
    parser.add_argument('--src', type=str, default='triviaqa', help='triviaqa or squad or qangaroo')

    parser.add_argument('--domain', type=str, default='web-open', help='domain web/wiki/web-open')

    parser.add_argument('--min_para_len', type=int, default=0,
                        help='Minimum length of a paragraph. Triviaqa has a lot of small paragraphs.')
    parser.add_argument('--max_para_len', type=int, default=400,
                        help='Maximum length of a paragraph. Ignore really long paragraphs too.')
    parser.add_argument('--max_train_questions', type=int, default=600000,
                        help='Maximum number of questions to train on. TriviaQA is huge.')
    parser.add_argument('--num_train_in_memory', type=int, default=50000,
                        help='Maximum number of questions to keep in memory at once.')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--embed_dir', type=str)
    parser.add_argument('--word_embeddings_file', type=str, default='data/embeddings/fasttext')
    parser.add_argument('--eval_file', type=str, default='web-dev.json')
    parser.add_argument('--verified_eval_file', type=str, default='verified-web-dev.json')
    parser.add_argument('--eval_only', type=int, default=0, help='Load a saved model and evaluate on dev set')
    parser.add_argument('--eval_correct_paras', type=int, default=0,
                        help='eval by sending only the correct paras of the doc')
    parser.add_argument('--train_correct_paras', type=int, default=0,
                        help='train a model on the correct paras of the doc')
    parser.add_argument('--eval_verified', type=int, default=1, help='eval the verified dev set after each partition.')

    # '--train_file', type = str, default = 'train.txt'
    parser.add_argument('--train_file_name', type=str, default='processed_train')
    parser.add_argument('--dev_file_name', type=str, default='processed_dev')
    parser.add_argument('--test_file_name', type=str, default='processed_test')

    parser.add_argument('--embedding_file', type=str, default='crawl-300d-2M.txt')
    parser.add_argument('--embedding_table', type=str, default='embedding_table.mdl')

    parser.add_argument('--para_mode', type=int, default=1,
                        help='represent a doc as a list of paras instead of a huge list of words')
    parser.add_argument('--small', type=int, default=0,
                        help='small dataset')

    parser.add_argument('--eval_only_para_clf', type=int, default=0, help='Load a saved model and evaluate on dev set')
    parser.add_argument('--pretrained_words', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--neg_sample', type=float, default=1.0)
    parser.add_argument('--test', type=int, default=0)

    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--use_tfidf_retriever', type=int, default=0, help='An additional tf-idf retriever to weed out paras')
    parser.add_argument('--num_topk_paras', type=int, default=5, help='Number of paras to choose from')
    parser.add_argument('--save_para_clf_output', type=int, default=0,
                        help='Save the top-k para returned by the para classifier')
    parser.add_argument('--save_para_clf_output_dir', type=str, default=None,
                        help='Path where to save')


    parser.add_argument('--pretrained', type=str, default=None, help='Pre-trained model')

    parser.add_argument('--use_qemb',
        type='bool',
        default=True,
        help='Whether to use weighted question embeddings'
    )
    parser.add_argument(
        '--use_in_question',
        type='bool',
        default=True,
        help='Whether to use in_question features'
    )
    parser.add_argument(
        '--use_pos',
        type='bool',
        default=False,
        help='Whether to use pos features'
    )
    parser.add_argument(
        '--use_ner',
        type='bool',
        default=False,
        help='Whether to use ner features'
    )
    parser.add_argument(
        '--use_lemma',
        type='bool',
        default=False,
        help='Whether to use lemma features'
    )
    parser.add_argument(
        '--use_tf',
        type='bool',
        default=False,
        help='Whether to use tf features'
    )
    parser.add_argument(
        '--unlabeled',
        type='bool',
        default=False,
        help='Data is unlabeled (prediction only)'
    )
    parser.add_argument(
        '--use_distant_supervision',
        type='bool',
        default=True,
        help='Whether to gather labels by distant supervision'
    )
    parser.add_argument(
        '--use_single_answer_alias',
        type='bool',
        default=False,
        help='Whether to use one alias of the answer i.e. just use "Obama" or all aliases for "Obama"'
    )
    parser.add_argument(
        '--fix_embeddings',
        type='bool',
        default=True,
        help='Keep word embeddings fixed (pretrained)'
    )
    parser.add_argument(
        '--paraclf_hidden_size', type=int, default=300, help='Hidden size of paragraph classifier',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of epochs (default 40)'
    )
    parser.add_argument(
        '--display_iter',
        type=int,
        default=25,
        help='Print train error after every \
                                 <display_iter> epoches (default 25)'
    )
    parser.add_argument(
        '--dropout_emb',
        type=float,
        default=0.1,
        help='Dropout rate for word embeddings'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamax',
        help='Optimizer: sgd or adamax (default)'
    )
    parser.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        default=0.1,
        help='Learning rate for SGD (default 0.1)'
    )
    parser.add_argument(
        '--grad_clipping',
        type=float,
        default=10,
        help='Gradient clipping (default 10.0)'
    )
    parser.add_argument(
        '--use_annealing_schedule',
        type='bool',
        default=True,
        help='Whether to use an annealing schedule or not.'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='Weight decay (default 0)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0, help='Momentum (default 0)'
    )
    args = parser.parse_args()

    if len(args.embedding_file) == 0:
        args.embedding_file = None
    else:
        args.embedding_file = os.path.join(args.word_embeddings_file, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)
    args.embedding_table = os.path.join(args.data_dir, args.src,  "embeddings",  args.domain,  args.embedding_table)
    args.final_model_dir = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    args.model_dir = os.path.join(args.model_dir, args.final_model_dir)
    args.log_file = os.path.join(args.model_dir, 'log.txt')
    args.model_file = os.path.join(args.model_dir, 'model.mdl')
    args.para_mode = (args.para_mode == 1)
    args.eval_only_para_clf = (args.eval_only_para_clf == 1)
    args.eval_verified = (args.eval_verified == 1)
    args.use_tfidf_retriever = (args.use_tfidf_retriever == 1)
    args.pretrained_words = (args.pretrained_words == 1)
    args.use_only_distant_sup = (args.use_only_distant_sup == 1)

    args.vocab_dir = os.path.join(args.data_dir , args.src,  'vocab', args.domain +"/")
    if not os.path.exists(args.vocab_dir+'tok2ind.json'):
        args.create_vocab = True

    subprocess.call(['mkdir', '-p', args.model_dir])

    # Embeddings options
    if args.embedding_file is not None:
        with smart_open(args.embedding_file) as f:
            dim = len(f.readline().decode('utf-8').strip().split(' ')) - 1
        args.embedding_dim = dim
    elif args.embedding_dim is None:
        raise RuntimeError(
            'Either embedding_file or embedding_dim '
            'needs to be specified.'
        )

    return args