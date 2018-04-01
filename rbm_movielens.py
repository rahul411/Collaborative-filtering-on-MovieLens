import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math

max_ratings = 5

def create_user_item_matrix():
    ratings = pd.read_csv(os.path.join('ml-1m/', 'ratings.dat'), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['userid', 'movieid', 'rating', 'timestamp'])

    users = pd.read_csv(os.path.join('ml-1m/', 'users.dat'), 
                        sep='::', 
                        engine='python', 
                        encoding='latin-1',
                        names=['userid', 'gender', 'age', 'occupation', 'zipcode'])

    movies = pd.read_csv(os.path.join('ml-1m/', 'movies.dat'), 
                        sep='::',  
                        encoding='latin-1',
                        engine = 'python',
                        names=['movieid', 'title', 'genre'])

    user_item_matrix = ratings.pivot(index = 'userid', columns ='movieid', values = 'rating').fillna(value=0)
    user_item_matrix = user_item_matrix.as_matrix()
    return user_item_matrix

def create_input_data():
    user_item_matrix = create_user_item_matrix()
    user_item_matrix = user_item_matrix[:10,:10]
    print(user_item_matrix)
    no_users , no_items = user_item_matrix.shape
    input_data = np.zeros([no_users,no_items,max_ratings])

    for i in range(no_users):
        for j in range(no_items):
            if user_item_matrix[i][j] == 0:
                input_data[i][j] = np.zeros(max_ratings)
            else:
                input_data[i][j][int(user_item_matrix[i][j])-1] = 1
    input_data = np.reshape(input_data,[input_data.shape[0],-1]).astype(dtype=np.float32)
    return input_data

class RBM():

    def __init__(self, input_data):
        self.model_path = 'model/'
        self.hidden_dim = 100
        self.visible_dim = input_data.shape[1]
        self.stddev = 1.0
        self.learning_rate = 0.001
        self.batchsize = 10
        self.alpha = 1
        self.epochs = 50
        self.training_data = input_data
        self.K = 10

        self.W = tf.get_variable('weights', shape = [self.visible_dim, self.hidden_dim], 
            initializer = tf.truncated_normal_initializer(stddev=self.stddev/math.sqrt(float(self.visible_dim))))
        self.h_bias = tf.get_variable('h_bias', shape=[self.hidden_dim],initializer= tf.constant_initializer(0))
        self.v_bias = tf.get_variable('v_bias', shape=[self.visible_dim],initializer= tf.constant_initializer(0))

    def sample_distribution(self,prob):
        return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))


    def sample_h_given_v(self,v):
        h_prob = tf.nn.sigmoid(tf.matmul(v,self.W) + self.h_bias)
        h_sample = self.sample_distribution(h_prob)
        return h_sample, h_prob

    def sample_v_given_h(self, h):
        v_sample = tf.matmul(h, tf.transpose(self.W)) + self.v_bias
        # v_mask = tf.sign(self.train_batch)
        v_sample = tf.reshape(v_sample,[tf.shape(v_sample)[0],-1,max_ratings])
        v_prob = tf.nn.softmax(v_sample)
        v_prob = tf.reshape(v_prob,[tf.shape(v_sample)[0],-1])
        v_sample = self.sample_distribution(v_prob)

        return v_sample, v_prob

    def gibbs_step(self,v):
        positive = 0
        negative = 0
        h_sample = 0
        h_prob0 = 0
        v_prob = 0
        v_sample = 0
        for step in range(self.K):
            h_sample, h_prob = self.sample_h_given_v(v)
            if step == 0:
                h_prob0 = h_prob
                positive = tf.matmul(tf.transpose(v*self.mask),h_sample)

            v_sample, v_prob = self.sample_v_given_h(h_sample)
            h_sample, h_prob = self.sample_h_given_v(v_prob)
            v = v_prob
        negative = tf.matmul(tf.transpose(v*self.mask),h_sample)
        return positive, negative, h_prob0, h_prob, v_prob, v_sample

    def create_graph(self):
        self.X = tf.placeholder(tf.float32, [None, self.visible_dim])
        self.mask = self.X
        self.mask = tf.reshape(self.mask,[self.batchsize,-1,max_ratings])
        self.mask = tf.reduce_sum(self.mask,axis=2)
        self.mask = tf.expand_dims(self.mask,axis=2)
        self.mask = tf.tile(self.mask,[1,1,5])
        self.mask = tf.reshape(self.mask,[self.batchsize,-1])

        positive, negative, h_prob0, h_prob, v_prob, v_sample = self.gibbs_step(self.X)
        w_gradient = self.learning_rate*(positive-negative)
        h_bias_gradient = tf.reduce_mean(self.learning_rate*(h_prob0-h_prob), axis=0)
        v_bias_gradient = tf.reduce_mean(self.learning_rate*(self.X*self.mask - v_prob*self.mask), axis=0)

        w_update = self.W.assign_add(w_gradient)
        h_bias_update = tf.assign_add(self.h_bias, h_bias_gradient)
        v_bias_update = tf.assign_add(self.v_bias, v_bias_gradient)

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.X*self.mask,v_sample*self.mask))
        tf.summary.scalar('loss',self.loss)
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.run_update = [w_update, h_bias_update, v_bias_update]

        self.v_prob_distribution = v_prob
        self.gradients = [w_gradient,h_bias_gradient,v_bias_gradient]
        # return self.run_update, self.loss, self.summary, self.v, w_gradient

    def train(self):
        # config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter("output", sess.graph)
        self.create_graph()
        for epoch in range(self.epochs):
            for start, end in zip(range(0,len(self.training_data),self.batchsize), range(self.batchsize,len(self.training_data)+1,self.batchsize)):
                updates, loss, _summary, v_dist, all_gradients = sess.run([self.run_update,self.loss,self.summary,self.v_prob_distribution,self.gradients], feed_dict={self.X: self.training_data[start:end]})
                # print("gradient bro:",np.mean(abs(all_gradients[0])))
                print(loss)
                print(v_dist[0,:10])
                writer.add_summary(_summary)
            print('End of epoch')        
            
            if epoch%10 == 0:
                self.saver.save(sess, os.path.join(self.model_path, 'model'), global_step=epoch)
                writer.flush()
        writer.close()


training_data = create_input_data()
print(training_data.shape)
# training_data = np.ones([100,1000], dtype=np.float32)
print('Done loading data')
rbm = RBM(training_data)
rbm.train()

