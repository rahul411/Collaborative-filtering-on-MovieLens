import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
from sklearn.metrics import mean_squared_error
from utils import load_dataset

max_ratings = 5
train_path = 'ml-100k/u1.base'
test_path = 'ml-100k/u1.test'
separator = '\t'

#This function is for creating user_item_matrix for  movieLens-100k
def form_user_item_matrix():
    all_users, all_movies, train_data, test_data = load_dataset(train_path, test_path, separator)
    train_user_item_matrix = np.zeros((max(all_users), max(all_movies)))
    test_user_item_matrix = np.zeros((max(all_users), max(all_movies)))

    for key, values in train_data.items():
        for item, rating in values:
            train_user_item_matrix[key-1,item-1] = rating 

    for key, values in test_data.items():
        for item, rating in values:
            test_user_item_matrix[key-1,item-1] = rating

    return train_user_item_matrix, test_user_item_matrix
    
#This function is for creating user_item_matrix for movieLens-1M
def create_user_item_matrix():
    ratings = pd.read_csv(os.path.join('ml-1m/', 'ratings.dat'), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['userid', 'movieid', 'rating', 'timestamp'])

    user_item_matrix = ratings.pivot(index = 'userid', columns ='movieid', values = 'rating').fillna(value=0)
    user_item_matrix = user_item_matrix.as_matrix()
    return user_item_matrix

def create_input_data(user_item_matrix):
    rated=0.0
    unrated=0.0

    no_users , no_items = user_item_matrix.shape
    input_data = np.zeros([no_users,no_items,max_ratings])

    for i in range(no_users):
        for j in range(no_items):
            if user_item_matrix[i][j] == 0:
                input_data[i][j] = np.zeros(max_ratings)
                unrated+=1
            else:
                input_data[i][j][int(user_item_matrix[i][j])-1] = 1
                rated+=1
    input_data = np.reshape(input_data,[input_data.shape[0],-1]).astype(dtype=np.float32)
    
    return input_data, rated, unrated

class RBM():

    def __init__(self, input_data):
        self.model_path = 'model/'
        self.hidden_dim = 2000
        self.no_users = input_data.shape[0]
        self.visible_dim = input_data.shape[1]
        self.stddev = 1.0
        self.learning_rate = 0.0005
        self.batchsize = 10
        self.alpha = 1
        self.epochs = 50
        self.momentum = 0.1
        self.input_data = input_data
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

        w_momentum_update = self.W*self.momentum + w_gradient 
        h_momentum_update = self.h_bias*self.momentum + h_bias_gradient
        v_momentum_update = self.v_bias*self.momentum + v_bias_gradient

        w_update = self.W.assign_add(w_momentum_update)
        h_bias_update = tf.assign_add(self.h_bias, h_momentum_update)
        v_bias_update = tf.assign_add(self.v_bias, v_momentum_update)

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.X*self.mask,v_sample*self.mask))
        tf.summary.scalar('loss',self.loss)
        self.summary = tf.summary.merge_all()

        self.run_update = [w_update, h_bias_update, v_bias_update]

        self.v_prob_distribution = v_prob
        self.predicted_ratings = v_sample
        self.train_predictions = v_sample*self.mask
        self.gradients = [w_gradient,h_bias_gradient,v_bias_gradient]

    def train(self, sess, rated, unrated):
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter("output", sess.graph)
        self.create_graph()
        self.saver = tf.train.Saver()
        for epoch in range(self.epochs):
            results = []
            test_preds = []
            for start, end in zip(range(0,len(self.input_data),self.batchsize), range(self.batchsize,len(self.input_data)+1,self.batchsize)):
                updates, loss, _summary, v_dist, all_gradients, train_preds, preds = sess.run([self.run_update,self.loss,self.summary,self.v_prob_distribution,self.gradients, self.train_predictions, self.predicted_ratings], feed_dict={self.X: self.input_data[start:end]})
                
                results.append(train_preds)
                test_preds.append(preds)
                
                writer.add_summary(_summary)
            print('End of epoch:', epoch)        
            
            if epoch%7 == 0:
                self.saver.save(sess, os.path.join(self.model_path, 'model'), global_step=epoch)
                writer.flush()
                # self.learning_rate = self.learning_rate/2 
            train_predictions = np.concatenate(results,axis=0)
            rmse = math.sqrt((unrated/rated+1)*mean_squared_error(np.argmax(np.reshape(self.input_data[:train_predictions.shape[0]],(train_predictions.shape[0],-1,max_ratings)),axis=2),np.argmax(np.reshape(train_predictions,(train_predictions.shape[0],-1,max_ratings)),axis=2)))
            print('Final rmse:',rmse)
        test_preds = np.concatenate(test_preds,axis=0)
        self.saver.save(sess, os.path.join(self.model_path, 'model'), global_step=epoch)
        # np.save('predictions.npy',test_preds)
        writer.close()

    def test(self, sess, rated, unrated, model_path):

        init = tf.global_variables_initializer()
        sess.run(init)
        self.create_graph()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        results = []
        test_preds = []
        for start, end in zip(range(0,len(self.input_data),self.batchsize), range(self.batchsize,len(self.input_data)+1,self.batchsize)):
            v_dist, train_preds, preds = sess.run([self.v_prob_distribution, self.train_predictions, self.predicted_ratings], feed_dict={self.X: self.input_data[start:end]})
            
            results.append(train_preds)
            test_preds.append(preds)

        test_predictions = np.concatenate(results,axis=0)
        rmse = math.sqrt((unrated/rated+1)*mean_squared_error(np.argmax(np.reshape(self.input_data[:test_predictions.shape[0]],(test_predictions.shape[0],-1,max_ratings)),axis=2),np.argmax(np.reshape(test_predictions,(test_predictions.shape[0],-1,max_ratings)),axis=2)))
        print('Final rmse:',rmse)
        test_preds = np.concatenate(test_preds,axis=0)
        
        np.save('predictions.npy',test_preds)


training_data, testing_data = form_user_item_matrix()
training_data, rated, unrated = create_input_data(training_data)
# training_data = np.ones([100,1000], dtype=np.float32)
print('Done loading data')
sess = tf.InteractiveSession()

#TODO: Currently both training and testing can't be done in the same run. This would require code modification, which will be fixed later.
# rbm = RBM(training_data)
# rbm.train(sess, rated, unrated)

testing_data, rated, unrated = create_input_data(testing_data)
rbm = RBM(testing_data)
rbm.test(sess, rated, unrated, 'model/model-7')
