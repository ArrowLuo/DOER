from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset
from general_utils import get_logger
from model import Model
from config import Config
from itertools import chain
from build_data import build_data
import argparse

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_preprocess', default=False, action='store_true')
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_evaluate', default=False, action='store_true')
    parser.add_argument('--current_path', type=str, default=".")
    parser.add_argument('--dim_domain', type=int, default=100, help='dimension size of domain-specific embedding.')
    parser.add_argument('--dim_general', type=int, default=300, help='dimension size of general embedding.')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate for optimizer.')
    parser.add_argument('--batch_size', type=int, default=16, help='the training batch size.')
    parser.add_argument('--nepochs', type=int, default=200, help='the max training epoch.')
    parser.add_argument('--nepoch_no_imprv', type=int, default=10, help='the coefficient of stop early.')
    parser.add_argument('--dropout', type=float, default=0.55, help='the dropout coefficient.')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='the coefficient of decayed learning rate.')
    parser.add_argument('--data_sets', type=str, default="laptops_2014", help="dataset for train, dev, and test.")
    parser.add_argument('--use_mpqa', default=False, action='store_true', help='auxilary sentimet lexicon enhancement.')
    parser.add_argument('--use_labels_length', default=False, action='store_true', help='auxilary sentimet lexicon enhancement.')
    parser.add_argument('--choice_rnncell', type=str, default="mgru", help='choice the rnn cell type.', choices=['mgru'])
    parser.add_argument('--do_cross_share', default=False, action='store_true')
    parser.add_argument('--cross_share_k', type=int, default=5, help='dimension size of paramenter `k` in cross share unit.')
    parser.add_argument('--show_test_results', default=False, action='store_true')
    parser.add_argument('--show_process_logs', default=False, action='store_true')

    args, _ = parser.parse_known_args()

    return args

def config_from_args(args):
    config = Config()
    for key, value in vars(args).items():
        config.__dict__[key] = value
    config.auto_config()
    logger = get_logger(config.log_path)
    return config, logger

if __name__ == "__main__":
    args = parse_parameters()
    config, logger = config_from_args(args)
    if args.do_preprocess:
        build_data(config, logger)

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_poss = load_vocab(config.poss_filename)
    vocab_chunks = load_vocab(config.chunks_filename)
    vocab_aspect_tags = load_vocab(config.aspect_tags_filename)
    vocab_polarity_tags = load_vocab(config.polarity_tags_filename)
    vocab_joint_tags = load_vocab(config.joint_tags_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, lowercase=config.lowercase)
    processing_pos = get_processing_word(vocab_poss, lowercase=False)
    processing_chunk = get_processing_word(vocab_chunks, lowercase=False)
    processing_aspect_tag = get_processing_word(vocab_aspect_tags, lowercase=False)
    processing_polarity_tag = get_processing_word(vocab_polarity_tags, lowercase=False)
    processing_joint_tag = get_processing_word(vocab_joint_tags, lowercase=False)

    # get pre trained embeddings
    domain_embeddings = get_trimmed_glove_vectors(config.domain_trimmed_filename)
    general_embeddings = get_trimmed_glove_vectors(config.general_trimmed_filename)

    # create dataset
    dev = CoNLLDataset(config.dev_filename, processing_word, processing_pos, processing_chunk, processing_aspect_tag,
                       processing_polarity_tag, processing_joint_tag, config.max_iter)
    test = CoNLLDataset(config.test_filename, processing_word, processing_pos, processing_chunk, processing_aspect_tag,
                        processing_polarity_tag, processing_joint_tag, config.max_iter)
    train = CoNLLDataset(config.train_filename, processing_word, processing_pos, processing_chunk,
                         processing_aspect_tag, processing_polarity_tag, processing_joint_tag, config.max_iter)

    data = [dev, test, train]
    _no_use_ = map(len, chain.from_iterable(w for w in (s for s in data)))
    max_sentence_size = max(train.max_sentence_len, test.max_sentence_len, dev.max_sentence_len)

    # build model
    config.n_aspect_tags = len(vocab_aspect_tags)
    config.n_polarity_tags = len(vocab_polarity_tags)
    config.n_joint_tags = len(vocab_joint_tags)
    config.n_poss = len(vocab_poss)
    config.n_chunks = len(vocab_chunks)
    config.n_words = len(vocab_words)
    config.max_sentence_size = max_sentence_size
    model = Model(config, domain_embeddings, general_embeddings, logger=logger,)
    model.build()

    if args.do_train:
        model.train(train, dev, vocab_words, vocab_poss, vocab_chunks, vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags)

    if args.do_evaluate:
        model.evaluate(test, vocab_words, vocab_poss, vocab_chunks, vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags)