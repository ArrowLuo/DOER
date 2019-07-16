import os
import tensorflow as tf
import numpy as np
import math

from data_utils import minibatches_for_sequence, pad_sequences, get_chunks, get_polaity_chunks, labels_average_length
from general_utils import Progbar
import general_utils as logging
from mgru_cell import MgRUCell

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model(object):
    def __init__(self, config, domain_embeddings, general_embeddings, logger=None, graph_suffix=None, ):
        """
        Args:
                config: class with hyper parameters
                domain_embeddings: np array with domain_embeddings
                general_embeddings: np array with general_embeddings
                logger: logger instance
        """
        self.config = config
        self.domain_embeddings = domain_embeddings
        self.general_embeddings = general_embeddings

        if graph_suffix is None:
            graph_suffix = '0'
        self.graph_suffix = str(graph_suffix)

        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        self.logger = logger

    def add_placeholders(self):
        """
        Adds placeholders to self
        """
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, self.config.max_sentence_size], name="word_ids" + self.graph_suffix)

        # shape = (batch size, max length of sentence in batch)
        self.pos_ids = tf.placeholder(tf.int32, shape=[None, self.config.max_sentence_size], name="pos_ids" + self.graph_suffix)

        # shape = (batch size, max length of sentence in batch)
        # chunk_ids is used to replace mpqa sometimes
        self.chunk_ids = tf.placeholder(tf.int32, shape=[None, self.config.max_sentence_size], name="chunk_ids" + self.graph_suffix)

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths" + self.graph_suffix)

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths" + self.graph_suffix)

        # shape = (batch size, max length of sentence in batch)
        self.labels_aspect = tf.placeholder(tf.int32, shape=[None, None], name="labels_aspect" + self.graph_suffix)
        self.labels_aspect_average_length = tf.placeholder(tf.float32, shape=[None], name="labels_aspect_average_length" + self.graph_suffix)

        # shape = (batch size, max length of sentence in batch)
        self.labels_polarity = tf.placeholder(tf.int32, shape=[None, None], name="labels_polarity" + self.graph_suffix)

        # shape = (batch size, max length of sentence in batch)
        self.labels_joint = tf.placeholder(tf.int32, shape=[None, None], name="labels_joint" + self.graph_suffix)

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout" + self.graph_suffix)
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr" + self.graph_suffix)

    def get_feed_dict(self, words, poss, chunks, labels_aspect=None, labels_polarity=None, labels_joint=None, lr=None,
                      dropout=None, vocab_aspect_tags=None):
        """
        Given some data, pad it and build a feed dictionary
        Args:
                words: list of sentences. A sentence is a list of ids of a list of words.
                        A word is a list of ids
                poss: list of poss_ids
                chunks: list of chunks_ids
                labels_aspect: list of labels_aspect_ids
                lr: (float) learning rate
                dropout: (float) keep prob
        Returns:
                dict {placeholder: value}
        """
        # perform padding of the given data
        word_ids, sequence_lengths = pad_sequences(words, self.config.n_words, self.config.max_sentence_size)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if poss is not None:
            poss, _ = pad_sequences(poss, self.config.n_poss, self.config.max_sentence_size)
            feed[self.pos_ids] = poss

        if chunks is not None:
            if self.config.use_mpqa:
                chunks, _ = pad_sequences(chunks, 0, self.config.max_sentence_size)
            else:
                chunks, _ = pad_sequences(chunks, self.config.n_chunks, self.config.max_sentence_size)
            feed[self.chunk_ids] = chunks

        if self.config.use_labels_length:
            if labels_aspect is not None and vocab_aspect_tags is not None:
                labels_average_ = labels_average_length(labels_aspect, vocab_aspect_tags)
                feed[self.labels_aspect_average_length] = labels_average_

        if labels_aspect is not None:
            labels_aspect, _ = pad_sequences(labels_aspect, 0, self.config.max_sentence_size)
            feed[self.labels_aspect] = labels_aspect

        if labels_polarity is not None:
            labels_polarity, _ = pad_sequences(labels_polarity, 0, self.config.max_sentence_size)
            feed[self.labels_polarity] = labels_polarity

        if labels_joint is not None:
            labels_joint, _ = pad_sequences(labels_joint, 0, self.config.max_sentence_size)
            feed[self.labels_joint] = labels_joint

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self, task_name):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope("words" + task_name + self.graph_suffix):
            nil_word_slot = np.ndarray(shape=(1, self.domain_embeddings.shape[-1]), dtype=np.float32, buffer=np.zeros([1, self.domain_embeddings.shape[-1]]))
            _embeddings = np.concatenate((self.domain_embeddings, nil_word_slot), axis=0)
            _domain_embeddings = tf.Variable(_embeddings, name="_domain_embeddings" + task_name + self.graph_suffix, dtype=tf.float32, trainable=self.config.train_embeddings)
            domain_embeddings = tf.nn.embedding_lookup(_domain_embeddings, self.word_ids, name="domain_embeddings" + task_name + self.graph_suffix)

            nil_word_slot = np.ndarray(shape=(1, self.general_embeddings.shape[-1]), dtype=np.float32, buffer=np.zeros([1, self.general_embeddings.shape[-1]]))
            _embeddings = np.concatenate((self.general_embeddings, nil_word_slot), axis=0)
            _general_embeddings = tf.Variable(_embeddings, name="_general_embeddings" + task_name + self.graph_suffix, dtype=tf.float32, trainable=self.config.train_embeddings)
            general_embeddings = tf.nn.embedding_lookup(_general_embeddings, self.word_ids, name="general_embeddings" + task_name + self.graph_suffix)

            x_emb = tf.concat([general_embeddings, domain_embeddings], axis=-1)
            x_emb = tf.nn.dropout(x_emb, self.dropout)

        return x_emb

    def add_mgru_layer(self, task_name, word_embeddings, hidden_size):
        with tf.variable_scope("bi-mgru" + task_name + self.graph_suffix):
            mgru_cell_f = MgRUCell(hidden_size)
            mgru_cell_b = MgRUCell(hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                mgru_cell_f, mgru_cell_b, word_embeddings, sequence_length=self.sequence_lengths, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)  # shape = (?,?,600)
            word_embeddings_output = tf.nn.dropout(output, self.dropout)
        return word_embeddings_output

    def add_crossdiff_op(self, aspect_hidden, polarity_hidden, m_hidden, g_hidden_size, max_pooling=True):
        with tf.variable_scope("crossdiff_variable_"):
            init_narray = np.random.randn(g_hidden_size, m_hidden, g_hidden_size)
            G_aspect_polarity = tf.Variable(init_narray, name="G_aspect_polarity", dtype=tf.float32)
            init_narray = np.random.randn(g_hidden_size, m_hidden, g_hidden_size)
            G_polarity_aspect = tf.Variable(init_narray, name="G_polarity_aspect", dtype=tf.float32)

            init_narray = np.random.randn(m_hidden, 1)
            G_vector_aspect = tf.Variable(init_narray, name="G_vector_aspect", dtype=tf.float32)
            init_narray = np.random.randn(m_hidden, 1)
            G_vector_polarity = tf.Variable(init_narray, name="G_vector_polarity", dtype=tf.float32)

        with tf.variable_scope("crossdiff_op_"):
            # Get share representation
            G_aspect_polarity = tf.reshape(G_aspect_polarity, shape=[g_hidden_size, -1])
            G_aspect_polarity = tf.tile(tf.expand_dims(G_aspect_polarity, axis=0), multiples=tf.stack([tf.shape(aspect_hidden)[0], 1, 1]))
            shared_hidden_aspect_polarity = tf.matmul(aspect_hidden, G_aspect_polarity)
            shared_hidden_aspect_polarity = tf.reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size * m_hidden, g_hidden_size])
            polarity_hidden_transpose = tf.transpose(polarity_hidden, [0, 2, 1])
            shared_hidden_aspect_polarity = tf.tanh(tf.matmul(shared_hidden_aspect_polarity, polarity_hidden_transpose))
            shared_hidden_aspect_polarity = tf.reshape(shared_hidden_aspect_polarity, [-1, self.config.max_sentence_size, m_hidden, self.config.max_sentence_size])
            if max_pooling:
                shared_hidden_aspect_polarity = tf.reduce_max(shared_hidden_aspect_polarity, axis=-2)
            else:
                shared_hidden_aspect_polarity = tf.transpose(shared_hidden_aspect_polarity, [0, 1, 3, 2])
                shared_hidden_aspect_polarity = tf.reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size * self.config.max_sentence_size, m_hidden])
                G_vector_aspect = tf.tile(tf.expand_dims(G_vector_aspect, axis=0), multiples=tf.stack([tf.shape(aspect_hidden)[0], 1, 1]))
                shared_hidden_aspect_polarity = tf.matmul(shared_hidden_aspect_polarity, G_vector_aspect)
            aspect_vector = tf.reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size, self.config.max_sentence_size])

            G_polarity_aspect = tf.reshape(G_polarity_aspect, shape=[g_hidden_size, -1])
            G_polarity_aspect = tf.tile(tf.expand_dims(G_polarity_aspect, axis=0), multiples=tf.stack([tf.shape(polarity_hidden)[0], 1, 1]))
            shared_hidden_polarity_aspect = tf.matmul(polarity_hidden, G_polarity_aspect)
            shared_hidden_polarity_aspect = tf.reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size * m_hidden, g_hidden_size])
            aspect_hidden_transpose = tf.transpose(aspect_hidden, [0, 2, 1])
            shared_hidden_polarity_aspect = tf.tanh(tf.matmul(shared_hidden_polarity_aspect, aspect_hidden_transpose))
            shared_hidden_polarity_aspect = tf.reshape(shared_hidden_polarity_aspect, [-1, self.config.max_sentence_size, m_hidden, self.config.max_sentence_size])
            if max_pooling:
                shared_hidden_polarity_aspect = tf.reduce_max(shared_hidden_polarity_aspect, axis=-2)
            else:
                shared_hidden_polarity_aspect = tf.transpose(shared_hidden_polarity_aspect, [0, 1, 3, 2])
                shared_hidden_polarity_aspect = tf.reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size * self.config.max_sentence_size, m_hidden])
                G_vector_polarity = tf.tile(tf.expand_dims(G_vector_polarity, axis=0), multiples=tf.stack([tf.shape(polarity_hidden)[0], 1, 1]))
                shared_hidden_polarity_aspect = tf.matmul(shared_hidden_polarity_aspect, G_vector_polarity)
            polarity_vector = tf.reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size, self.config.max_sentence_size])

            # Get attention vector
            aspect_attention_vector = tf.nn.softmax(aspect_vector, dim=-1)
            polarity_attention_vector = tf.nn.softmax(polarity_vector, dim=-1)

            aspect_hidden_v = tf.matmul(aspect_attention_vector, polarity_hidden)
            polarity_hidden_v = tf.matmul(polarity_attention_vector, aspect_hidden)

            aspect_hidden = aspect_hidden + aspect_hidden_v
            polarity_hidden = polarity_hidden + polarity_hidden_v

            aspect_hidden = tf.reshape(aspect_hidden, shape=[-1, self.config.max_sentence_size, g_hidden_size])
            polarity_hidden = tf.reshape(polarity_hidden, shape=[-1, self.config.max_sentence_size, g_hidden_size])

        return aspect_hidden, polarity_hidden, aspect_attention_vector, polarity_attention_vector

    def build(self):

        self.add_placeholders()

        word_embeddings = self.add_word_embeddings_op('_extraction_')
        embeddings_one, embeddings_two = word_embeddings, word_embeddings

        if self.config.choice_rnncell == "mgru":
            embeddings_one = self.add_mgru_layer('_embedding_one_layer1_', embeddings_one, self.config.hidden_size)
            embeddings_two = self.add_mgru_layer('_embedding_two_layer1_', embeddings_two, self.config.hidden_size)
        first_aspect_hidden, first_polarity_hidden = embeddings_one, embeddings_two

        if self.config.do_cross_share:
            with tf.variable_scope('crossdiff'+str(1)+'_'):
                cross_share_k, g_hidden_size = self.config.cross_share_k, 2 * self.config.hidden_size
                max_pooling = False
                embeddings_one, embeddings_two, self.aspect_attv, self.polarity_attv = self.add_crossdiff_op(
                    embeddings_one, embeddings_two, cross_share_k, g_hidden_size, max_pooling=max_pooling)

        if self.config.choice_rnncell == "mgru":
            embeddings_one = self.add_mgru_layer('_embedding_one_layer2_', embeddings_one, self.config.hidden_size)
            embeddings_two = self.add_mgru_layer('_embedding_two_layer2_', embeddings_two, self.config.hidden_size)

        aspect_hidden, polarity_hidden = embeddings_one, embeddings_two

        if self.config.use_mpqa:
            self.mpqa_logits = self.add_full_connection_op('_mpqa_extraction_', first_polarity_hidden, self.config.n_chunks, active_function=None)

        if self.config.crf_loss:
            self.aspect_logits = self.add_full_connection_op('_aspect_extraction_', aspect_hidden, self.config.n_aspect_tags, active_function=tf.nn.relu)
            self.polarity_logits = self.add_full_connection_op('_polarity_extraction_', polarity_hidden, self.config.n_polarity_tags, active_function=tf.nn.relu)
        else:
            self.aspect_logits = self.add_full_connection_op('_aspect_extraction_', aspect_hidden, self.config.n_aspect_tags, active_function=None) # tf.nn.softmax
            self.polarity_logits = self.add_full_connection_op('_polarity_extraction_', polarity_hidden, self.config.n_polarity_tags, active_function=None)

        if self.config.use_labels_length:
            aspect_length_hidden = tf.reduce_max(first_aspect_hidden, axis=-2)
            aspect_length_logits = self.add_full_connection_op('_aspect_length_', aspect_length_hidden, 1, active_function=tf.sigmoid)
            self.aspect_length_logits = tf.reshape(aspect_length_logits, [-1])

            polarity_length_hidden = tf.reduce_max(first_polarity_hidden, axis=-2)
            polarity_length_logits = self.add_full_connection_op('_polarity_length_', polarity_length_hidden, 1, active_function=tf.sigmoid)
            self.polarity_length_logits = tf.reshape(polarity_length_logits, [-1])

        self.aspect_pred = self.add_pred_op('_extraction_', self.aspect_logits)
        self.polarity_pred = self.add_pred_op('_extraction_', self.polarity_logits)
        self.add_loss_op('_extraction_')
        self.add_train_op('_extraction_')
        self.add_init_op()

    def add_full_connection_op(self, task_name, inputv, out_size, active_function=None):
        """
        """
        inputv_shape = inputv.get_shape().as_list()
        in_size = inputv_shape[-1]
        with tf.variable_scope("proj" + task_name + self.graph_suffix):
            stdv_ = 1. / math.sqrt(in_size)
            randomW_ = np.random.uniform(low=-stdv_, high=stdv_, size=(in_size, out_size))
            randomb_ = np.random.uniform(low=-stdv_, high=stdv_, size=(out_size))
            W = tf.Variable(randomW_, name="W", dtype=tf.float32)
            b = tf.Variable(randomb_, name="b", dtype=tf.float32)

            ntime_steps = inputv_shape[1]
            output = tf.reshape(inputv, [-1, in_size])
            pred = tf.matmul(output, W) + b
            if active_function is not None:
                pred = active_function(pred)
            if len(inputv_shape) == 2:
                logits = tf.reshape(pred, [-1, out_size])  # shape = (?,?,3)
            else:
                logits = tf.reshape(pred, [-1, ntime_steps, out_size])  # shape = (?,?,3)

        return logits

    def add_pred_op(self, task_name, logits):
        labels_pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        return labels_pred

    def add_loss_op(self, task_name):
        """
        Adds loss to self
        """
        self.aspect_transition_params = None
        self.polarity_transition_params = None
        label_length_loss = 0
        if self.config.crf_loss:
            # The first loss of aspect
            num_tags = self.aspect_logits.get_shape()[2].value
            stdv_ = 1. / math.sqrt(num_tags)
            randomW_ = np.random.uniform(low=-stdv_, high=stdv_, size=(num_tags, num_tags))
            aspect_transitions = tf.Variable(randomW_, name="aspect_transitions" + task_name + self.graph_suffix, dtype=tf.float32)
            log_likelihood, self.aspect_transition_params = tf.contrib.crf.crf_log_likelihood(
                self.aspect_logits, self.labels_aspect, self.sequence_lengths, aspect_transitions
            )
            aspect_loss = tf.reduce_mean(-log_likelihood)
            # The second loss of polarity
            num_tags = self.polarity_logits.get_shape()[2].value
            polarity_transitions = tf.Variable(np.random.randn(num_tags, num_tags),
                                               name="polarity_transitions" + task_name + self.graph_suffix,
                                               dtype=tf.float32) / np.sqrt(num_tags / 2)
            log_likelihood, self.polarity_transition_params = tf.contrib.crf.crf_log_likelihood(
                self.polarity_logits, self.labels_polarity, self.sequence_lengths, polarity_transitions
            )
            polarity_loss = tf.reduce_mean(-log_likelihood)
        else:
            # The first loss of aspect
            aspect_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.aspect_logits, labels=self.labels_aspect)
            mask = tf.sequence_mask(self.sequence_lengths, maxlen=self.config.max_sentence_size)
            aspect_loss = tf.boolean_mask(aspect_loss, mask)
            aspect_loss = tf.reduce_mean(aspect_loss)
            # The second loss of polarity
            polarity_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.polarity_logits, labels=self.labels_polarity)
            mask = tf.sequence_mask(self.sequence_lengths, maxlen=self.config.max_sentence_size)
            polarity_loss = tf.boolean_mask(polarity_loss, mask)
            polarity_loss = tf.reduce_mean(polarity_loss)

        self.loss = aspect_loss + polarity_loss

        if self.config.use_labels_length:
            aspect_label_length_loss = tf.losses.mean_squared_error(labels=self.labels_aspect_average_length, predictions=self.aspect_length_logits)
            polarity_label_length_loss = tf.losses.mean_squared_error(labels=self.labels_aspect_average_length, predictions=self.polarity_length_logits)
            self.label_length_loss = aspect_label_length_loss + polarity_label_length_loss
            self.loss = self.loss + self.label_length_loss

        if self.config.use_mpqa:
            mpqa_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.mpqa_logits, labels=self.chunk_ids)
            mask = tf.sequence_mask(self.sequence_lengths, maxlen=self.config.max_sentence_size)
            mpqa_loss = tf.boolean_mask(mpqa_loss, mask)
            self.mpqa_loss = tf.reduce_mean(mpqa_loss)
            self.loss = self.loss + self.mpqa_loss

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += 0.001 * sum(regularization_loss)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def add_train_op(self, task_name):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step" + task_name + self.graph_suffix):
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)

            global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.lr, global_step, self.config.decay_steps, self.config.lr_decay, staircase=True)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)  # , beta1=0.9
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)

    def train(self, train, dev, vocab_words, vocab_poss, vocab_chunks, vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags):
        """
        Performs training with early stopping and lr exponential decay
        Args:
                train: dataset that yields tuple of sentences, tags
                dev: dataset
                vocab_words: {word: index} dictionary
                vocab_poss: {pos: index} dictionary
                vocab_chunks: {chunk: index} dictionary
                tags: {tag: index} dictionary
        """
        best_score = 10000
        reinit_sess = 20
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0

        gpuConfig = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        gpuConfig.gpu_options.allow_growth = True
        with tf.Session(config=gpuConfig) as sess:
            sess.run(self.init)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

                aspect_p, aspect_r, aspect_f1, train_loss, \
                polarity_p, polarity_r, polarity_f1, dev_loss = self.run_epoch(sess, train, dev, vocab_words, vocab_poss, vocab_chunks,
                                                             vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags, epoch)

                if self.config.data_sets.startswith("twitter"):
                    dev_loss = train_loss
                # early stopping and saving best parameters
                current_score = dev_loss
                if train_loss > reinit_sess:
                    sess.run(self.init)
                if current_score < best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = current_score
                    self.logger.info("- new best score %.4f !" % (best_score))
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                        break

    def run_epoch(self, sess, train, dev, vocab_words, vocab_poss, vocab_chunks,
                  vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
                sess: tensorflow session
                train: dataset that yields tuple of sentences, tags
                dev: dataset
                vocab_aspect_tags: {tag: index} dictionary
                epoch: (int) number of the epoch
        """
        self.config.istrain = True  # set to train first, #batch normalization#
        losses = []
        nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (words, poss, chunks, labels_aspect, labels_polarity, labels_joint) in enumerate(
                minibatches_for_sequence(train, self.config.batch_size)):
            fd, sequence_lengths = self.get_feed_dict(words, poss, chunks, labels_aspect, labels_polarity, labels_joint,
                                                      self.config.lr, self.config.dropout, vocab_aspect_tags=vocab_aspect_tags)

            _, lr, train_loss, summary = sess.run([self.train_op, self.learning_rate, self.loss, self.merged], feed_dict=fd)
            losses.append(train_loss)

            if self.config.show_process_logs:
                print_mess = [("train loss", train_loss)]
                if self.config.use_labels_length:
                    label_length_loss = sess.run(self.label_length_loss, feed_dict=fd)
                    print_mess.append(("label_length_loss", label_length_loss))
                if self.config.use_mpqa:
                    mpqa_loss = sess.run(self.mpqa_loss, feed_dict=fd)
                    print_mess.append(("mpqa_loss", mpqa_loss))
                print_mess.append(("lr", lr))

                prog.update(i + 1, print_mess)

            # tensorboard
            if i % 2 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        if self.config.data_sets.startswith("twitter"):
            aspect_p, aspect_r, aspect_f1, aspect_test_acc, \
            polarity_p, polarity_r, polarity_f1, polarity_test_acc, dev_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0
            self.logger.info("Ignore validating without corresponding data~")
        else:
            aspect_p, aspect_r, aspect_f1, aspect_test_acc, \
            polarity_p, polarity_r, polarity_f1, polarity_test_acc, dev_loss = self.run_evaluate(
                sess, dev, vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags, vocab_words, is_dev=True)

            self.logger.info("- aspect_dev precision {:04.2f} - aspect_dev recall {:04.2f} - "
                             "aspect_dev f1 {:04.2f} - aspect_dev acc {:04.2f}".format
                             (100 * aspect_p, 100 * aspect_r, 100 * aspect_f1, 100 * aspect_test_acc))
            self.logger.info("- polarity_dev precision {:04.2f} - polarity_dev recall {:04.2f} - "
                             "polarity_dev f1 {:04.2f} - polarity_dev acc {:04.2f}".format
                             (100 * polarity_p, 100 * polarity_r, 100 * polarity_f1, 100 * polarity_test_acc))

        return aspect_p, aspect_r, aspect_f1, sum(losses) / len(losses), polarity_p, polarity_r, polarity_f1, dev_loss

    # These formulas come from SemEval 2014 Task 4
    def cacul_f1(self, correct_preds, total_preds, total_correct):
        p = correct_preds / total_preds if total_preds > 0 else 0.
        r = correct_preds / total_correct if total_correct > 0 else 0.
        f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.
        return p, r, f1

    def get_aspect_polarity_pairs(self, asps, pols):
        asps_dict = {}
        for a in asps:
            key = a[1] * 1000 + a[2]
            if key not in asps_dict:
                asps_dict[key] = a[0]

        strs = []
        for b in pols:
            key = b[1] * 1000 + b[2]
            if key in asps_dict:
                strs.append(str(b[1]) + "-" + str(b[2]) + "-" + asps_dict[key] + "-" + b[0])
        return strs

    def run_evaluate(self, sess, test, vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags, vocab_words, is_dev=True):
        """
        Evaluates performance on test set
        """
        self.config.istrain = False  # set to test first, #batch normalization#
        idx_to_words = {}
        if self.config.show_test_results:
            idx_to_words = {idx: word for word, idx in vocab_words.iteritems()}
        losses = []
        aspect_test_accs, polarity_test_accs = [], []
        aspect_correct_preds, aspect_total_correct, aspect_total_preds = 0., 0., 0.
        polarity_correct_preds, polarity_total_correct, polarity_total_preds = 0., 0., 0.

        for words, poss, chunks, labels_aspect, labels_polarity, labels_joint in minibatches_for_sequence(test, self.config.test_batch_size):

            if self.config.show_test_results:
                if type(words) == tuple:
                    char_ids, word_ids = zip(*words)
                else:
                    char_ids, word_ids = [], words

            aspect_lab_chunks = []
            aspect_lab_pred_chunks = []
            # Just used to evaluate Aspect
            labels_pred, sequence_lengths = self.predict_batch(sess, words, poss, chunks, vocab_words,
                                                               self.aspect_logits, self.aspect_transition_params, self.aspect_pred)
            for lab, lab_pred, length in zip(labels_aspect, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                aspect_test_accs += map(lambda a_b: a_b[0] == a_b[1], zip(lab, lab_pred))

                lab_chunks = get_chunks(lab, vocab_aspect_tags)
                aspect_lab_chunks.append(lab_chunks)
                lab_chunks = set(lab_chunks)

                lab_pred_chunks = get_chunks(lab_pred, vocab_aspect_tags)
                aspect_lab_pred_chunks.append(lab_pred_chunks)
                lab_pred_chunks = set(lab_pred_chunks)

                aspect_correct_preds += len(lab_chunks & lab_pred_chunks)
                aspect_total_preds += len(lab_pred_chunks)
                aspect_total_correct += len(lab_chunks)

            # Just used to evaluate Polarity
            labels_pred, sequence_lengths = self.predict_batch(sess, words, poss, chunks, vocab_words,
                                                               self.polarity_logits, self.polarity_transition_params, self.polarity_pred)
            index = 0
            for lab, lab_pred, length in zip(labels_polarity, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                polarity_test_accs += map(lambda a_b: a_b[0] == a_b[1], zip(lab, lab_pred))

                lab_chunks = set(get_polaity_chunks(lab, vocab_polarity_tags, aspect_lab_chunks[index]))
                lab_pred_chunks = set(get_polaity_chunks(lab_pred, vocab_polarity_tags, aspect_lab_pred_chunks[index]))
                polarity_correct_preds += len(lab_chunks & lab_pred_chunks)
                polarity_total_preds += len(lab_pred_chunks)
                polarity_total_correct += len(lab_chunks)

                if self.config.show_test_results:
                    self.logger.info(" ".join([idx_to_words[w] for w in word_ids[index][:length]]))
                    self.logger.info("T: " + " ".join(self.get_aspect_polarity_pairs(aspect_lab_chunks[index], lab_chunks)))
                    self.logger.info("P: " + " ".join(self.get_aspect_polarity_pairs(aspect_lab_pred_chunks[index], lab_pred_chunks)))

                index += 1

            # get loss
            fd, sequence_lengths = self.get_feed_dict(words, poss, chunks, labels_aspect=labels_aspect, labels_polarity=labels_polarity, labels_joint=labels_joint, dropout=1.0, vocab_aspect_tags=vocab_aspect_tags)
            dev_loss = sess.run(self.loss, feed_dict=fd)
            losses.append(dev_loss)

        aspect_p, aspect_r, aspect_f1 = self.cacul_f1(aspect_correct_preds, aspect_total_preds, aspect_total_correct)
        aspect_test_acc = np.mean(aspect_test_accs)

        polarity_p, polarity_r, polarity_f1 = self.cacul_f1(polarity_correct_preds, polarity_total_preds, polarity_total_correct)
        polarity_test_acc = np.mean(polarity_test_accs)

        return aspect_p, aspect_r, aspect_f1, aspect_test_acc, polarity_p, polarity_r, polarity_f1, polarity_test_acc, sum(losses)/len(losses)

    def evaluate(self, test, vocab_words, vocab_poss, vocab_chunks, vocab_aspect_tags, vocab_polarity_tags,
                 vocab_joint_tags):
        saver = tf.train.Saver()

        gpuConfig = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        gpuConfig.gpu_options.allow_growth = True
        with tf.Session(config=gpuConfig) as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.config.model_output)
            aspect_p, aspect_r, aspect_f1, aspect_test_acc, \
            polarity_p, polarity_r, polarity_f1, polarity_test_acc, test_loss = self.run_evaluate(
                sess, test, vocab_aspect_tags, vocab_polarity_tags, vocab_joint_tags, vocab_words, is_dev=False)
            self.logger.info("- aspect_test precision {:04.2f} - aspect_test recall {:04.2f} - "
                             "aspect_test f1 {:04.2f} - aspect_test acc {:04.2f}".format
                             (100 * aspect_p, 100 * aspect_r, 100 * aspect_f1, 100 * aspect_test_acc))
            self.logger.info("- polarity_test precision {:04.2f} -polarity_test recall {:04.2f} - "
                             "polarity_test f1 {:04.2f} - polarity_test acc {:04.2f}".format
                             (100 * polarity_p, 100 * polarity_r, 100 * polarity_f1, 100 * polarity_test_acc))

    def predict_batch(self, sess, words, poss, chunks, vocab_words, trained_logits, trained_transition_params, labels_pred):
        """
        Args:
                sess: a tensorflow session
                words: list of sentences
        Returns:
                labels_pred: list of labels for each sentence
                sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(words, poss, chunks, dropout=1.0)
        if self.config.crf_loss:
            pred_sequences = []
            logits, transition_params = sess.run([trained_logits, trained_transition_params], feed_dict=fd)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
                pred_sequences += [viterbi_sequence]
        else:
            pred_sequences = sess.run(labels_pred, feed_dict=fd)

        return pred_sequences, sequence_lengths
