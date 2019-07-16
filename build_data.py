import os
from data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    write_vocab, load_vocab, \
    export_trimmed_glove_vectors, get_processing_word

def build_data(config, logger):
    """
    Procedure to build data
    """
    processing_word = get_processing_word(lowercase=config.lowercase)

    # Generators
    test = CoNLLDataset(config.test_filename, processing_word)
    dev = CoNLLDataset(config.dev_filename, processing_word)
    train = CoNLLDataset(config.train_filename, processing_word)

    # Build Word and Tag vocab
    print("Build Word and Tag vocab...")
    vocab_words, vocab_poss, vocab_chunks, \
    vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags = get_vocabs([train, dev, test])
    vocab = vocab_words
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    print("Dealing words vocab...")
    write_vocab(vocab, config.words_filename)
    print("Dealing poss vocab...")
    write_vocab(vocab_poss, config.poss_filename)

    vocab_chunks = [tags for tags in vocab_chunks]
    if "NO" in vocab_chunks:
        vocab_chunks.remove("NO")
        vocab_chunks.insert(0, "NO")
    else:
        logger.error(">>> vocab_chunks used as mpqa has something wrong!")
    print("Dealing chunks vocab...")
    write_vocab(vocab_chunks, config.chunks_filename)

    vocab_aspect_tags = [tags for tags in vocab_aspect_tags]
    vocab_aspect_tags.remove("O")
    vocab_aspect_tags.insert(0, "O")
    vocab_polarity_tags = [tags for tags in vocab_polarity_tags]
    vocab_polarity_tags.remove("O")
    vocab_polarity_tags.insert(0, "O")
    vocab_joint_tags = [tags for tags in vocab_joint_tags]
    vocab_joint_tags.remove("O")
    vocab_joint_tags.insert(0, "O")
    print("Dealing aspect_tags vocab...")
    write_vocab(vocab_aspect_tags, config.aspect_tags_filename)
    print("Dealing polarity_tags vocab...")
    write_vocab(vocab_polarity_tags, config.polarity_tags_filename)
    print("Dealing joint_tags vocab...")
    write_vocab(vocab_joint_tags, config.joint_tags_filename)

    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.domain_filename, config.domain_trimmed_filename, config.dim_domain)
    export_trimmed_glove_vectors(vocab, config.general_filename, config.general_trimmed_filename, config.dim_general)
