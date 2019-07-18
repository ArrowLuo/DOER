import os
class Config(object):
    def __init__(self):
        self.dim_domain = 100
        self.dim_general = 300
        self.data_sets = "laptops_2014"

        # model control
        self.choice_rnncell = "regu"
        self.do_cross_share = False
        self.cross_share_k = 5
        self.use_mpqa = False
        self.use_labels_length = False
        self.show_test_results = False
        self.show_process_logs = True

        # default
        self.crf_loss = True
        self.train_embeddings = False
        self.current_path = "."
        # data
        self.max_iter = None
        self.lowercase = False

        self.nepochs = 200
        self.dropout = 0.55
        self.batch_size = 16
        self.lr = 0.001
        self.lr_decay = 0.95
        self.decay_steps = 500
        self.nepoch_no_imprv = 10

        self.test_batch_size = 256
        self.hidden_size = 300
        self.char_hidden_size = 100

        self.istrain = True  # train for default, this value will update in model

        # derivative variable
        self.n_aspect_tags = 0
        self.n_polarity_tags = 0
        self.n_joint_tags = 0
        self.n_poss = 0
        self.n_chunks = 0
        self.n_words = 0
        self.max_sentence_size = 0
        self.max_word_size = None

    def auto_config(self):
        _amazon_filename = "{}/data/amazon/amazon_reviews_small.{}d.txt".format(self.current_path, self.dim_domain)
        _yelp_filename = "{}/data/yelp/yelp_reviews_small.{}d.txt".format(self.current_path, self.dim_domain)
        _amazon_trimmed_filename = "{}/data/model_data/amazon_reviews.{}.{}d.trimmed.npz".format(self.current_path, self.data_sets, self.dim_domain)
        _yelp_trimmed_filename = "{}/data/model_data/yelp_reviews.{}.{}d.trimmed.npz".format(self.current_path, self.data_sets, self.dim_domain)

        self.domain_filename = _amazon_filename
        self.domain_trimmed_filename = _amazon_trimmed_filename
        self.general_filename = "{}/data/glove.840B/glove_small.840B.{}d.txt".format(self.current_path, self.dim_general)
        self.general_trimmed_filename = "{}/data/model_data/glove.840B.{}.{}d.trimmed.npz".format(self.current_path, self.data_sets, self.dim_general)

        data_sets_name = self.data_sets.split("_")[0]
        assert data_sets_name in ['laptops', 'restaurants', 'twitter']

        if data_sets_name == 'laptops':
            self.domain_filename = _amazon_filename
            self.domain_trimmed_filename = _amazon_trimmed_filename
        elif data_sets_name == 'restaurants':
            self.domain_filename = _yelp_filename
            self.domain_trimmed_filename = _yelp_trimmed_filename
        elif data_sets_name == 'twitter':
            self.domain_filename = _amazon_filename
            self.domain_trimmed_filename = _amazon_trimmed_filename

        self.words_filename = "{}/data/model_data/words_{}.txt".format(self.current_path, self.data_sets)
        self.poss_filename = "{}/data/model_data/poss_{}.txt".format(self.current_path, self.data_sets)
        self.chunks_filename = "{}/data/model_data/chunk_{}.txt".format(self.current_path, self.data_sets)
        self.aspect_tags_filename = "{}/data/model_data/aspect_tags_{}.txt".format(self.current_path, self.data_sets)
        self.polarity_tags_filename = "{}/data/model_data/polarity_tags_{}.txt".format(self.current_path, self.data_sets)
        self.joint_tags_filename = "{}/data/model_data/joint_tags_{}.txt".format(self.current_path, self.data_sets)

        self.test_filename = "{}/data/{}/{}_test_mpqa.gold.txt".format(self.current_path, data_sets_name, self.data_sets)
        self.dev_filename = "{}/data/{}/{}_trial_mpqa.txt".format(self.current_path, data_sets_name, self.data_sets)
        self.train_filename = "{}/data/{}/{}_train_mpqa.txt".format(self.current_path, data_sets_name, self.data_sets)

        output_root = "{}/results".format(self.current_path)
        self.output_path = "{}/{}/".format(output_root, self.data_sets)
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"

        model_data_path = "{}/data/model_data/".format(self.current_path)
        output_path_path = self.output_path
        model_output_path = self.model_output
        if os.path.exists(output_root) is False:
            os.mkdir(output_root)
        if os.path.exists(model_data_path) is False:
            os.mkdir(model_data_path)
        if os.path.exists(output_path_path) is False:
            os.mkdir(output_path_path)
        if os.path.exists(model_output_path) is False:
            os.mkdir(model_output_path)
