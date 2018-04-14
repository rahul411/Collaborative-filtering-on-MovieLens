from __future__ import division, print_function

import tensorflow as tf
import numpy as np


class VAECF(object):
    """docstring for VAECF"""
    def __init__(self, sess, num_user, num_item, hidden_encoder_dim, hidden_decoder_dim, latent_dim, learning_rate,
                batch_size, k, beta, reg_param, one_hot=False, user_embed_dim = 500, item_embed_dim = 500):
        
        self.sess = sess
        self.num_item = num_item
        self.num_user = num_user
        self.hidden_encoder_dim = hidden_encoder_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate    
        self.reg_param = reg_param
        self.one_hot = one_hot
        self.user_embed_dim = user_embed_dim
        self.item_embed_dim = item_embed_dim
        self.k = k
        self.beta = beta

        if self.one_hot:
            self.user_input_dim = self.num_user   #6040
            self.item_input_dim = self.num_item   #3952
        else:
            self.user_input_dim = self.num_item    #3952
            self.item_input_dim = self.num_user    #6040

        self.build_model()

    def build_model(self):

        self.l2_loss = tf.constant(0.0)

        self.user = tf.placeholder("float", shape=[None, self.user_input_dim])   # 1x3952
        self.item = tf.placeholder("float", shape=[None, self.item_input_dim])   # 1x6040

        self.user_idx = tf.placeholder(tf.int64, shape=[None])
        self.item_idx = tf.placeholder(tf.int64, shape=[None])

        self.rating = tf.placeholder("float", shape=[None])

        if self.one_hot:

            self.W_user_embed = tf.Variable(tf.truncated_normal([self.user_input_dim, self.user_embed_dim], stddev = 0.01), 
                                name = 'user_embed_weights')
            self.W_item_embed = tf.Variable(tf.truncated_normal([self.item_input_dim, self.item_embed_dim], stddev = 0.01), 
                                name = 'item_embed_weights')


            self.W_encoder_input_hidden_user = tf.Variable(tf.truncated_normal([self.user_embed_dim, self.hidden_encoder_dim], 
                                stddev = 0.01), name = 'W_encoder_input_hidden_user')
            self.b_encoder_input_hidden_user = tf.get_variable('b_encoder_input_hidden_user', [self.hidden_encoder_dim],
                                                initializer=tf.constant_initializer(0.))


            self.W_encoder_input_hidden_item = tf.Variable(tf.truncated_normal([self.item_embed_dim, self.hidden_encoder_dim], 
                                stddev = 0.01), name = 'W_encoder_input_hidden_item')
            self.b_encoder_input_hidden_item = tf.get_variable('b_encoder_input_hidden_item', [self.hidden_encoder_dim],
                                                initializer=tf.constant_initializer(0.))

            self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.W_item_embed)
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.W_item_embed)

        else:

            self.W_encoder_input_hidden_user = tf.Variable(tf.truncated_normal([self.user_input_dim, self.hidden_encoder_dim], 
                                stddev = 0.01), name = 'W_encoder_input_hidden_user')            
            self.b_encoder_input_hidden_user = tf.get_variable('b_encoder_input_hidden_user', [self.hidden_encoder_dim],
                                                initializer=tf.constant_initializer(0.))


            self.W_encoder_input_hidden_item = tf.Variable(tf.truncated_normal([self.item_input_dim, self.hidden_encoder_dim], 
                                stddev = 0.01), name = 'W_encoder_input_hidden_item')
            self.b_encoder_input_hidden_item = tf.get_variable('b_encoder_input_hidden_item', [self.hidden_encoder_dim],
                                                initializer = tf.constant_initializer(0.))


        self.l2_loss += tf.nn.l2_loss(self.W_encoder_input_hidden_user)
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_input_hidden_item)


        # Hidden encoder layer

        if self.one_hot:
            self.user_embed = tf.nn.embedding_lookup(
                self.W_user_embed, self.user_idx)
            self.item_embed = tf.nn.embedding_lookup(
                self.W_item_embed, self.item_idx)
            self.hidden_encoder_user = tf.nn.relu(tf.matmul(
                self.user_embed, self.W_encoder_input_hidden_user) + self.b_encoder_input_hidden_user)
            self.hidden_encoder_item = tf.nn.relu(tf.matmul(
                self.item_embed, self.W_encoder_input_hidden_item) + self.b_encoder_input_hidden_item)

        else:

            self.hidden_encoder_user = tf.nn.relu(tf.matmul(
                self.user, self.W_encoder_input_hidden_user) + self.b_encoder_input_hidden_user)
            self.hidden_encoder_item = tf.nn.relu(tf.matmul(
                self.item, self.W_encoder_input_hidden_item) + self.b_encoder_input_hidden_item)


        self.W_encoder_hidden_mu_user = tf.Variable(tf.truncated_normal([self.hidden_encoder_dim, self.latent_dim], 
                                stddev = 0.01), name = 'W_encoder_hidden_mu_user')
        self.b_encoder_hidden_mu_user = tf.get_variable('b_encoder_hidden_mu_user', [self.latent_dim],
                                                initializer=tf.constant_initializer(0.))
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_mu_user)


        self.W_encoder_hidden_mu_item = tf.Variable(tf.truncated_normal([self.hidden_encoder_dim, self.latent_dim], 
                                stddev = 0.01), name = 'W_encoder_hidden_mu_item')
        self.b_encoder_hidden_mu_item = tf.get_variable('b_encoder_hidden_mu_item', [self.latent_dim],
                                                initializer=tf.constant_initializer(0.))
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_mu_item)


        # Mu encoder

        self.mu_encoder_user = tf.matmul(
            self.hidden_encoder_user, self.W_encoder_hidden_mu_user) + self.b_encoder_hidden_mu_user
        self.mu_encoder_item = tf.matmul(
            self.hidden_encoder_item, self.W_encoder_hidden_mu_item) + self.b_encoder_hidden_mu_item

        self.W_encoder_hidden_logvar_user = tf.Variable(tf.truncated_normal([self.hidden_encoder_dim, self.latent_dim], 
                                stddev = 0.01), name = 'W_encoder_hidden_logvar_user')
        self.b_encoder_hidden_logvar_user = tf.get_variable('b_encoder_hidden_logvar_user', [self.latent_dim],
                                                initializer=tf.constant_initializer(0.))

        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_logvar_user)

        self.W_encoder_hidden_logvar_item = tf.Variable(tf.truncated_normal([self.hidden_encoder_dim, self.latent_dim], 
                                stddev = 0.01), name = 'W_encoder_hidden_logvar_item')
        self.b_encoder_hidden_logvar_item = tf.get_variable('b_encoder_hidden_logvar_item', [self.latent_dim],
                                                initializer=tf.constant_initializer(0.))

        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_logvar_item)


        # Sigma encoder

        self.logvar_encoder_user = tf.matmul(
            self.hidden_encoder_user, self.W_encoder_hidden_logvar_user) + self.b_encoder_hidden_logvar_user
        self.logvar_encoder_item = tf.matmul(
            self.hidden_encoder_item, self.W_encoder_hidden_logvar_item) + self.b_encoder_hidden_logvar_item

        # Sample epsilon

        self.epsilon_user = tf.random_normal(
            tf.shape(self.logvar_encoder_user), name='epsilon_user')
        self.epsilon_item = tf.random_normal(
            tf.shape(self.logvar_encoder_item), name='epsilon_item')



        # Sample latent variable
        self.std_encoder_user = tf.exp(0.5 * self.logvar_encoder_user)
        self.z_user = self.mu_encoder_user + tf.multiply(self.std_encoder_user, self.epsilon_user)

        self.std_encoder_item = tf.exp(0.5 * self.logvar_encoder_item)
        self.z_item = self.mu_encoder_item + tf.multiply(self.std_encoder_item, self.epsilon_item)


        self.W_encoder_latent = tf.Variable(tf.truncated_normal([self.latent_dim, self.latent_dim], 
                                stddev = 0.01), name = 'weighted_inner_product')

        self.rating_pred = tf.multiply(tf.matmul(self.z_user, self.W_encoder_latent), self.z_item)


        # Decoder network

        self.W_decoder_z_hidden_user = tf.Variable(tf.truncated_normal([self.latent_dim, self.hidden_decoder_dim],
                                    stddev = 0.01), name = 'W_decoder_z_hidden_user')
        self.b_decoder_z_hidden_user = tf.get_variable('b_decoder_z_hidden_user', [self.hidden_decoder_dim],
                                            initializer=tf.constant_initializer(0.))

        # self.W_decoder_z_hidden_item = tf.Variable(tf.truncated_normal([self.latent_dim, self.hidden_decoder_dim],
        #                             stddev = 0.01), name = 'W_decoder_z_hidden_item')
        # self.b_decoder_z_hidden_item = tf.get_variable('b_decoder_z_hidden_item', [self.hidden_decoder_dim],
        #                                     initializer = tf.constant_initializer(0.))

        self.l2_loss += tf.nn.l2_loss(self.W_decoder_z_hidden_user)


        # Hidden layer decoder

        self.hidden_decoder_user = tf.nn.relu(tf.matmul(self.rating_pred, self.W_decoder_z_hidden_user) + self.b_decoder_z_hidden_user)

        self.W_decoder_hidden_reconstruction_user = tf.Variable(tf.truncated_normal([self.hidden_decoder_dim, self.user_input_dim],
                                    stddev = 0.01), name = 'W_decoder_hidden_reconstruction_user')
        self.b_decoder_hidden_reconstruction_user = tf.get_variable('b_decoder_hidden_reconstruction_user', [self.user_input_dim],
                                            initializer=tf.constant_initializer(0.))

        self.l2_loss += tf.nn.l2_loss(self.W_decoder_hidden_reconstruction_user)

        self.reconstructed_user = tf.matmul(
            self.hidden_decoder_user, self.W_decoder_hidden_reconstruction_user) + self.b_decoder_hidden_reconstruction_user

        weight = tf.not_equal(self.user, tf.constant(0, dtype=tf.float32))
        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.user, self.reconstructed_user, weight))
        self.reconstructed_user = tf.cast(self.reconstructed_user, tf.int64)
        # self.MAE = tf.reduce_mean(
        #     tf.abs(tf.subtract(self.user, self.reconstructed_user)))
        # self.Recall = tf.metrics.recall(tf.cast(self.user, tf.int64), self.reconstructed_user, weight)

        self.vals, self.indxs = tf.nn.top_k(self.reconstructed_user, k = 20, sorted=True)
        # self.user_vals = tf.gather(self.user, self.indxs)

        # self.Recall = tf.metrics.recall(tf.cast(self.user_vals, tf.int64), self.vals[0])
        self.Recall = tf.metrics.recall_at_top_k(tf.cast(self.user, tf.int64), self.indxs, k = 20)



        # Compute KL Divergence between prior p(z) and q(z|x)

        self.KLD = -0.5 * tf.reduce_sum(1 + self.logvar_encoder_user - tf.pow(
            self.mu_encoder_user, 2) - tf.exp(self.logvar_encoder_user), reduction_indices=1)

        self.KLD = self.KLD - 0.5 * tf.reduce_sum(1 + self.logvar_encoder_item - tf.pow(
            self.mu_encoder_item, 2) - tf.exp(self.logvar_encoder_item), reduction_indices=1)


        # Prediction

        # self.user_bias = tf.get_variable('user_bias', [self.num_user], initializer=tf.constant_initializer(0.))
        # self.item_bias = tf.get_variable('item_bias', [self.num_item], initializer=tf.constant_initializer(0.))

        # self.W_encoder_latent = tf.Variable(tf.truncated_normal([self.latent_dim, self.latent_dim], 
                                # stddev = 0.01), name = 'weighted_inner_product')

        # self.rating_pred = tf.reduce_sum(tf.multiply(tf.matmul(self.z_user, self.W_encoder_latent), self.z_item), 
        #                     reduction_indices=1)
        # self.rating_pred = tf.add(self.rating_pred, tf.nn.embedding_lookup(self.user_bias, self.user_idx))
        # self.rating_pred = tf.add(self.rating_pred, tf.nn.embedding_lookup(self.item_bias, self.item_idx))

        # self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.rating, self.rating_pred))))
        # self.Recall = tf.metrics.recall_at_k(self.rating, self.rating_pred, self.k)

        self.loss = tf.reduce_mean(self.RMSE + self.beta * self.KLD)
        self.regularized_loss = self.loss + self.reg_param * self.l2_loss

        # tf.summary.scalar("RMSE", self.RMSE)
        # tf.summary.scalar("RECALL", self.Recall)
        # tf.summary.scalar("Loss", self.loss)
        # tf.summary.scalar("Reg-Loss", self.regularized_loss)


        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.regularized_loss)

        # add op for merging summary
        # self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver()

###############################################################################################################


    def construct_feeddict(self, user_idx, item_idx, M):
        if self.one_hot:
            feed_dict = {self.user_idx: user_idx, self.item_idx: item_idx,
                         self.rating: M[user_idx, item_idx]}
        else:
            feed_dict = {self.user: M[user_idx, :], self.item: M[
                :, item_idx].transpose(), self.user_idx:user_idx, self.item_idx:item_idx, self.rating: M[user_idx, item_idx]}
        return feed_dict




    def train_test_validation(self, data, train_idx, test_idx, valid_idx, n_steps=10000, result_path='result/'):

        nonzero_user_idx = data.nonzero()[0]
        nonzero_item_idx = data.nonzero()[1]

        train_size = train_idx.size
        train_data = np.zeros(data.shape)

        train_data[nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]] = data[nonzero_user_idx[train_idx], 
                                                                                    nonzero_item_idx[train_idx]]

        # train_writer = tf.summary.FileWriter(
        #     result_path + '/train', graph=self.sess.graph)
        # valid_writer = tf.summary.FileWriter(
        #     result_path + '/validation', graph=self.sess.graph)
        # test_writer = tf.summary.FileWriter(
        #     result_path + '/test', graph=self.sess.graph)


        best_val_rmse = np.inf
        best_test_rmse = 0
        best_val_recall = np.inf
        best_test_recall = 0

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        # self.sess.run(tf.global_variables_initializer())

        for step in range(1, n_steps):
                print(step)
                # self.sess.run(tf.global_variables_initializer())
                batch_idx = np.random.randint(train_size, size=self.batch_size)
                user_idx = nonzero_user_idx[train_idx[batch_idx]]
                item_idx = nonzero_item_idx[train_idx[batch_idx]]
                feed_dict = self.construct_feeddict(user_idx, item_idx, train_data)

                _, rmse, recall = self.sess.run(
                    [self.train_step, self.RMSE, self.Recall], feed_dict=feed_dict)

                # train_writer.add_summary(summary_str, step)

                if step % 100 == 0:

                    print ("training done. Validating...")
                    valid_user_idx = nonzero_user_idx[valid_idx]
                    valid_item_idx = nonzero_item_idx[valid_idx]
                    # valid_data = np.zeros(data.shape)
                    # valid_data[nonzero_user_idx[valid_user_idx], nonzero_item_idx[valid_item_idx]] = data[nonzero_user_idx[valid_user_idx], 
                    #                                                                 nonzero_item_idx[valid_item_idx]]
                    feed_dict = self.construct_feeddict(valid_user_idx, valid_item_idx, data)

                    rmse_valid, recall_valid = self.sess.run([self.RMSE, self.Recall], feed_dict=feed_dict)
                    # valid_writer.add_summary(summary_str, step)

                    test_user_idx = nonzero_user_idx[test_idx]
                    test_item_idx = nonzero_item_idx[test_idx]
                    # test_data = np.zeros(data.shape)
                    # test_data[nonzero_user_idx[test_user_idx], nonzero_item_idx[test_item_idx]] = data[nonzero_user_idx[test_user_idx], 
                    #                                                                 nonzero_item_idx[test_item_idx]]
                    feed_dict = self.construct_feeddict(test_user_idx, test_item_idx, data)

                    rmse_test, recall_test = self.sess.run([self.RMSE, self.Recall], feed_dict=feed_dict)
                    # test_writer.add_summary(summary_str, step)

                    print ("Step {0} | Train RMSE: {1:3.4f}, Train Recall: {2:3.4f}".format(
                        step, rmse, recall[0]))
                    print ("         | Valid  RMSE: {0:3.4f}, Valid Recall: {1:3.4f}".format(
                        rmse_valid, recall_valid[0]))
                    print ("         | Test  RMSE: {0:3.4f}, Test Recall: {1:3.4f}".format(
                        rmse_test, recall_test[0]))


                    if best_val_rmse > rmse_valid:
                        best_val_rmse = rmse_valid
                        best_test_rmse = rmse_test

                    if best_val_recall > recall_valid[0]:
                        best_val_recall = recall_valid[0]
                        best_test_recall = recall_test[0]


        self.saver.save(self.sess, result_path + "/model.ckpt")
        return best_test_rmse, best_test_recall

        