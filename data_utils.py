#!/usr/bin/python3
# -*- coding:utf8 -*-
import codecs

import numpy as np
import os
import math

# python version 2.*
import cPickle as pickle

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


class CoNLLDataset(object):
    """
    Class that iterates over CoNLL Dataset
    """

    def __init__(self, filename, processing_word=None, processing_pos=None, processing_chunk=None,
                 processing_aspect_tag=None, processing_polarity_tag=None, processing_joint_tag=None, max_iter=None):
        """
        Args:
            filename: path to the file
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_pos = processing_pos
        self.processing_chunk = processing_chunk
        self.processing_aspect_tag = processing_aspect_tag
        self.processing_polarity_tag = processing_polarity_tag
        self.processing_joint_tag = processing_joint_tag
        self.max_iter = max_iter
        self.length = None

        self.max_sentence_len = 0
        self.max_token_len = 0

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, poss, chunks, aspect_tags, polarity_tags, joint_tags = [], [], [], [], [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, poss, chunks, aspect_tags, polarity_tags, joint_tags
                        self.max_sentence_len = self.max_sentence_len if self.max_sentence_len > len(words) else len(words)
                        words, poss, chunks, aspect_tags, polarity_tags, joint_tags = [], [], [], [], [], []
                else:
                    word, pos, chunk, aspect_tag, polarity_tag, joint_tag = line.split(' ')
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                        if type(word) == tuple:
                            self.max_token_len = len(word[0]) if len(word[0]) > self.max_token_len else self.max_token_len
                    if self.processing_pos is not None:
                        pos = self.processing_pos(pos)
                    if self.processing_chunk is not None:
                        chunk = self.processing_chunk(chunk)
                    if self.processing_aspect_tag is not None:
                        aspect_tag = self.processing_aspect_tag(aspect_tag)
                    if self.processing_polarity_tag is not None:
                        polarity_tag = self.processing_polarity_tag(polarity_tag)
                    if self.processing_joint_tag is not None:
                        joint_tag = self.processing_joint_tag(joint_tag)
                    words += [word]
                    poss += [pos]
                    chunks += [chunk]
                    aspect_tags += [aspect_tag]
                    polarity_tags += [polarity_tag]
                    joint_tags += [joint_tag]

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    vocab_words = set()
    vocab_poss = set()
    vocab_chunks = set()
    vocab_aspect_tags = set()
    vocab_polarity_tags = set()
    vocab_joint_tags = set()
    for dataset in datasets:
        for words, poss, chunks, aspect_tags, polarity_tags, joint_tags in dataset:
            if type(words[0]) == tuple:
                words = zip(*words)[1]
            vocab_words.update(words)
            vocab_poss.update(poss)
            vocab_chunks.update(chunks)
            vocab_aspect_tags.update(aspect_tags)
            vocab_polarity_tags.update(polarity_tags)
            vocab_joint_tags.update(joint_tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_poss, vocab_chunks, vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags

def get_glove_vocab(filename, lowercase=False):
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            if lowercase:
                word = word.lower()
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx

    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    in_num = 0
    bingo_vocab = []
    stdv_ = 1. / math.sqrt(dim)
    embeddings = np.random.uniform(low=-stdv_, high=stdv_, size=(len(vocab), dim))

    # python 2.*
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(val) for val in line[1:]]
            if word in vocab and len(embedding) == dim:
                if word not in bingo_vocab:
                    bingo_vocab.append(word)
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)
                    in_num += 1

    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with open(filename, "rb") as f:
        return np.load(f)["embeddings"]


def get_processing_word(vocab_words=None, vocab_chars=None, lowercase=False, chars=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """

    def f(word):
        char_ids = []
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def labels_average_length(labels_aspect, vocab_aspect_tags):
    # [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
    length_list_ = []
    for labels in labels_aspect:
        chunks = get_chunks(labels, vocab_aspect_tags)
        length_chunks_list_ = []
        for ck in chunks:
            length_chunks_list_.append(ck[2] - ck[1])
        length_chunks_average = 0 if len(length_chunks_list_) == 0 else np.average(length_chunks_list_)
        length_list_.append(length_chunks_average)

    length_min = np.min(length_list_)
    length_max = np.max(length_list_)
    length_gap = length_max - length_min + 1e-8
    length_list_ = [2 * (x - length_min) / float(length_gap) - 1 for x in length_list_]  # [-1, 1]
    length_list_ = [1. / (1 + np.exp(-x)) for x in length_list_]
    return length_list_


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, fixed_sentence_length=None, fixd_words_length=None, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = fixed_sentence_length if fixed_sentence_length != None else max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = fixd_words_length if fixd_words_length != None else max(
            [max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = fixed_sentence_length if fixed_sentence_length != None else max(
            map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError("Paramenter `nlevels` is wrong.")

    return sequence_padded, sequence_length

def minibatches_for_sequence(seq_data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    # words, poss, chunks, labels_aspect, labels_polarity, labels_joint
    x_batch, y_batch, z_batch, u_batch, v_batch, w_batch = [], [], [], [], [], []
    for (x, y, z, u, v, w) in seq_data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, z_batch, u_batch, v_batch, w_batch
            x_batch, y_batch, z_batch, u_batch, v_batch, w_batch = [], [], [], [], [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
        z_batch += [z]
        u_batch += [u]
        v_batch += [v]
        w_batch += [w]

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch, u_batch, v_batch, w_batch


def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('_')[-1]


def get_chunk_alpha(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('_')[0]


def get_chunks(seq, vocab_tags):
    """
    Args:
        seq: [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1] sequence of labels
        vocab_tags: {'O': 0, 'B_AP': 1, 'I_AP': 2}
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1]
        vocab_tags = {'O': 0, 'B_AP': 1, 'I_AP': 2}
        result = [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
    """
    default = vocab_tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in vocab_tags.iteritems()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            tok_chunk_alpha = get_chunk_alpha(tok, idx_to_tag)
            if chunk_type is None and tok_chunk_alpha == "B":
                chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type is not None and tok_chunk_type != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
                if tok_chunk_alpha == "B":
                    chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type is not None and tok_chunk_type == chunk_type:
                if tok_chunk_alpha == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks


def get_polaity_chunks(seq, vocab_tags, aspect_lab_chunks):
    """
    Args:
        seq: [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1] sequence of labels
        vocab_tags: {'O': 0, 'POSITIVE': 1, 'NEUTRAL': 2, 'NEGATIVE':3, 'CONFLICT':4}
        aspect_lab_chunks: [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1]
        vocab_tags = {'O': 0, 'POSITIVE': 1, 'NEUTRAL': 2, 'NEGATIVE':3, 'CONFLICT':4}
        aspect_lab_chunks = [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
        result = [('POSITIVE', 0, 1), ('POSITIVE', 2, 3), ('POSITIVE', 3, 5), ('NEUTRAL', 6, 9), ('POSITIVE', 10, 11)]
    """
    idx_to_tag = {idx: tag for tag, idx in vocab_tags.iteritems()}
    default = vocab_tags[NONE]
    chunks = []
    for i, chunk in enumerate(aspect_lab_chunks):
        segs = seq[chunk[1]:chunk[2]]
        counts = np.bincount(np.array(segs, dtype=int))
        counts = np.where(counts == max(counts))
        for indx in counts[0]:
            if default != indx:
                chunk_type = idx_to_tag[int(indx)]
                chunk = (chunk_type, chunk[1], chunk[2])
                chunks.append(chunk)
    return chunks
