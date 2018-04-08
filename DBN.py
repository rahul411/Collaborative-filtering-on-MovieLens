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
X = {}
W = {}
h_bias = {}
v_bias = {}


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

    def __init__(self, visible_dim, hidden_dim, name):
        self.model_path = 'model/'
        self.hidden_dim = hidden_dim
        self.visible_dim = visible_dim
        self.stddev = 1.0
        self.learning_rate = 0.0005
        self.batchsize = 10
        self.alpha = 1
        self.epochs = 2
        self.momentum = 0.1
        self.name = name
        # self.input_data = input_data
        self.K = 10
        X[self.name] = tf.placeholder(tf.float32, [None, self.visible_dim], name='X'+name)
        W[self.name] = tf.get_variable(shape = [self.visible_dim, self.hidden_dim], 
            initializer = tf.truncated_normal_initializer(stddev=self.stddev/math.sqrt(float(self.visible_dim))), name='W'+name)
        h_bias[self.name] = tf.get_variable(shape=[self.hidden_dim],initializer= tf.constant_initializer(0), name='h_b'+name)
        v_bias[self.name] = tf.get_variable(shape=[self.visible_dim],initializer= tf.constant_initializer(0), name='v_b'+name)

    def sample_distribution(self,prob):
        return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))


    def sample_h_given_v(self,v):
        h_prob = tf.nn.sigmoid(tf.matmul(v,W[self.name]) + h_bias[self.name])
        h_sample = self.sample_distribution(h_prob)
        return h_sample, h_prob

    def sample_v_given_h(self, h):
        v_sample = tf.matmul(h, tf.transpose(W[self.name])) + v_bias[self.name]
        v_sample = tf.reshape(v_sample,[tf.shape(v_sample)[0],-1,max_ratings])
        v_prob = tf.nn.softmax(v_sample)
        v_prob = tf.reshape(v_prob,[tf.shape(v_sample)[0],-1])
        v_sample = self.sample_distribution(v_prob)

        return v_sample, v_prob

    def gibbs_step(self,v, mask):
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
                positive = tf.matmul(tf.transpose(v*mask),h_sample)

            v_sample, v_prob = self.sample_v_given_h(h_sample)
            h_sample, h_prob = self.sample_h_given_v(v_prob)
            v = v_prob
        negative = tf.matmul(tf.transpose(v*mask),h_sample)
        return positive, negative, h_prob0, h_prob, v_prob, v_sample

    def create_graph(self):
        # self.X = tf.placeholder(tf.float32, [None, self.visible_dim])
        mask = X[self.name]
        mask = tf.reshape(mask,[self.batchsize,-1,max_ratings])
        mask = tf.reduce_sum(mask,axis=2)
        mask = tf.expand_dims(mask,axis=2)
        mask = tf.tile(mask,[1,1,5])
        mask = tf.reshape(mask,[self.batchsize,-1])

        positive, negative, h_prob0, h_prob, v_prob, v_sample = self.gibbs_step(X[self.name], mask)
        w_gradient = self.learning_rate*(positive-negative)
        h_bias_gradient = tf.reduce_mean(self.learning_rate*(h_prob0-h_prob), axis=0)
        v_bias_gradient = tf.reduce_mean(self.learning_rate*(X[self.name]*mask - v_prob*mask), axis=0)

        w_momentum_update = W[self.name]*self.momentum + w_gradient 
        h_momentum_update = h_bias[self.name]*self.momentum + h_bias_gradient
        v_momentum_update = v_bias[self.name]*self.momentum + v_bias_gradient

        w_update = W[self.name].assign_add(w_momentum_update)
        h_bias_update = tf.assign_add(h_bias[self.name], h_momentum_update)
        v_bias_update = tf.assign_add(v_bias[self.name], v_momentum_update)


        self.run_update = [w_update, h_bias_update, v_bias_update]

        self.v_prob_distribution = v_prob
        self.predicted_ratings = v_sample
        self.train_predictions = v_sample*mask
        self.gradients = [w_gradient,h_bias_gradient,v_bias_gradient]

    def train(self, sess, input_data, rated, unrated):
        self.create_graph()
        for epoch in range(self.epochs):
            results = []
            test_preds = []
            for start, end in zip(range(0,len(input_data),self.batchsize), range(self.batchsize,len(input_data)+1,self.batchsize)):
                updates, v_dist, all_gradients, train_preds, preds = sess.run([self.run_update,
                                                                                self.v_prob_distribution,
                                                                                self.gradients, 
                                                                                self.train_predictions, 
                                                                                self.predicted_ratings], 
                                                                                feed_dict={X[self.name]: input_data[start:end]})
                
                results.append(train_preds)
                test_preds.append(preds)
                
            print('End of epoch for ' + self.name + ': '+ str(epoch))        
             
            train_predictions = np.concatenate(results,axis=0)
            rmse = math.sqrt((unrated/rated+1)*mean_squared_error(np.argmax(np.reshape(input_data[:train_predictions.shape[0]],(train_predictions.shape[0],-1,max_ratings)),axis=2),np.argmax(np.reshape(train_predictions,(train_predictions.shape[0],-1,max_ratings)),axis=2)))
            print('Final rmse for ' + self.name + ': '+ str(rmse))
        test_preds = np.concatenate(test_preds,axis=0)


class DBN():

    def __init__(self, no_hidden_layers, parameters):
        self.no_hidden_layers = no_hidden_layers
        self.rbms = []
        self.parameters = parameters

    def initialize_rbms(self):
        for i in range(self.no_hidden_layers):
            self.rbms.append(RBM(self.parameters[i][0], self.parameters[i][1], self.parameters[i][2]))

    def train(self, sess, training_data):
        input_data = training_data
        for k in range(self.no_hidden_layers):
            v = tf.convert_to_tensor(input_data)
            for i in range(k):
                h, _ = self.rbms[i].sample_h_given_v(v)
                v = h
            input_data = sess.run(v)
            self.rbms[k].train(sess, input_data, self.parameters[k][3], self.parameters[k][4])
        saver = tf.train.Saver()
        saver.save(sess, os.path.join('model/', 'model'), global_step=1)

    def test(self, sess, testing_data, rated, unrated, model_path):
        saver = tf.train.Saver()
        saver.restore(sess,model_path)
        no_users, no_items = testing_data.shape
        input_data = testing_data
        input_data = tf.convert_to_tensor(input_data)
        v = input_data
        for k in range(self.no_hidden_layers):
            h, _ = self.rbms[k].sample_h_given_v(v)
            v = h

        for k in range(self.no_hidden_layers-1,-1,-1):
            v, _ = self.rbms[k].sample_v_given_h(h)
            h = v

        mask = input_data
        mask = tf.reshape(mask,[no_users,-1,max_ratings])
        mask = tf.reduce_sum(mask,axis=2)
        mask = tf.expand_dims(mask,axis=2)
        mask = tf.tile(mask,[1,1,5])
        mask = tf.reshape(mask,[no_users,-1])

        v = v*mask
        test_preds = sess.run(v)
        rmse = math.sqrt((unrated/rated+1)*mean_squared_error(np.argmax(np.reshape(testing_data,(no_users,-1,max_ratings)),axis=2),
                                                                np.argmax(np.reshape(test_preds,(no_users,-1,max_ratings)),axis=2)))

        print('Final rmse', rmse)



training_data, testing_data = form_user_item_matrix()
training_data, train_rated, train_unrated = create_input_data(training_data)
# training_data = np.ones([100,1000], dtype=np.float32)
print('Done loading data')
sess = tf.InteractiveSession()

testing_data, test_rated, test_unrated = create_input_data(testing_data)

training_parameters = [(training_data.shape[1],2000,'hidden_1', train_rated, train_unrated),(2000,500,'hidden_2',1,1)]

dbn = DBN(2,training_parameters)
dbn.initialize_rbms()

init = tf.global_variables_initializer()
sess.run(init)
dbn.train(sess, training_data)
dbn.test(sess, testing_data, test_rated, test_unrated, 'model/model-1')


