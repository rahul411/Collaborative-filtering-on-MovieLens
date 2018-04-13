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

def get_weight_variable(name,shape):
    initializer = tf.contrib.layers.xavier_initializer()
    weight = tf.get_variable(name=name, dtype= tf.float32, shape=shape, initializer=initializer)
    return weight

def get_bias_variable(name,shape):
    bias = tf.Variable(tf.zeros(shape), name=name)
    return bias

class Autoencoder():
    def __init__(self, visible_dim):
        self.visible_dim = visible_dim
        # self.parameters = parameters
        self.learningRate = 0.001
        self.epochs = 100
        self.batchsize = 30
        
        self.W1_enc = get_weight_variable('encoder_W1', [visible_dim, 64])
        self.b1_enc = get_bias_variable('encoder_b1',[64])
        self.W2_enc = get_weight_variable('encoder_W2', [64, 32])
        self.b2_enc = get_bias_variable('encoder_b2',[32])
        
        self.W1_dec = get_weight_variable('decoder_W1', [32, 64])
        self.b1_dec = get_bias_variable('decoder_b1',[64])
        self.W2_dec = get_weight_variable('decoder_W2', [64, visible_dim])
        self.b2_dec = get_bias_variable('decoder_b2',[visible_dim])

    def encoder(self, input):
        h1 = tf.matmul(input,self.W1_enc) + self.b1_enc
        h1 = tf.nn.selu(h1)

        h2 = tf.matmul(h1,self.W2_enc) + self.b2_enc
        h2 = tf.nn.selu(h2)

        h3 = tf.nn.dropout(h2,0.5)
        return h3

    def decoder(self, encoder_out):
        h1 = tf.matmul(encoder_out, self.W1_dec) + self.b1_dec
        h1 = tf.nn.selu(h1)

        h2 = tf.matmul(h1, self.W2_dec) + self.b2_dec
        h2 = tf.nn.selu(h2)

        return h2

    def createGraph(self):
        self.X = tf.placeholder(tf.float32, [None, self.visible_dim])

        mask = tf.cast(self.X>0,dtype=tf.float32)
        rated = tf.count_nonzero(mask,dtype=tf.float32)
        unrated = tf.count_nonzero(tf.equal(mask,0), dtype=tf.float32)

        encoder_out = self.encoder(self.X)
        decoder_out = self.decoder(encoder_out)
        
        #Adjust the loss term according to the rated and unrated
        loss = (unrated/rated + 1)*tf.losses.mean_squared_error(self.X, mask*decoder_out)
        optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(loss)
        return loss, decoder_out, mask*decoder_out, optimizer

    def train(self, sess, input_data):

        loss, final_predictions, train_predictions, optimize = self.createGraph()        
        init = tf.global_variables_initializer()
        sess.run(init)

        self.saver = tf.train.Saver()
        for epoch in range(self.epochs):
            results = []
            test_preds = []
            for start, end in zip(range(0,len(input_data),self.batchsize), range(self.batchsize,len(input_data)+1,self.batchsize)):
                steploss, predictions, train_preds, _ = sess.run([loss, final_predictions, train_predictions, optimize], feed_dict={self.X: input_data[start:end]})
                # print('step1:',steploss)

                #This step is in accordance to the paper, f(f(x)) should be equal to f(x) ideally. f(x) is dense input. 
                steploss, predictions, train_preds, _ = sess.run([loss, final_predictions, train_predictions, optimize], feed_dict={self.X: predictions})
                # print('step2:',steploss)
                results.append(train_preds)
                test_preds.append(predictions)
              
            print('End of epoch:', epoch)        
                
            total_train_predictions = np.concatenate(results,axis=0)
            no_users, _ = total_train_predictions.shape
            rated = np.count_nonzero(input_data)
            unrated = np.count_nonzero(input_data==0)
            mask = input_data>0
            rmse = math.sqrt((unrated/rated+1)*mean_squared_error(input_data[:no_users], mask[:no_users]*total_train_predictions))
            print('Final rmse:',rmse)
        test_preds = np.concatenate(test_preds,axis=0)
        # self.saver.save(sess, os.path.join(self.model_path, 'model'), global_step=epoch)
        # np.save('predictions_autoencoders.npy',test_preds)

    def test(self, sess, input_data, model_path=None):

        loss, final_predictions, train_predictions, optimize = self.createGraph() 
        # init = tf.global_variables_initializer()
        # sess.run(init)
        
        # saver = tf.train.Saver()
        # saver.restore(sess, model_path)
        test_preds = []
        for start, end in zip(range(0,len(input_data),self.batchsize), range(self.batchsize,len(input_data)+1,self.batchsize)):
            preds = sess.run(final_predictions, feed_dict={self.X: input_data[start:end]})
            
            test_preds.append(preds)

        test_predictions = np.concatenate(test_preds,axis=0)
        rated = np.count_nonzero(input_data)
        unrated = np.count_nonzero(input_data==0)
        mask = input_data>0
        rmse = math.sqrt((unrated/rated+1)*mean_squared_error(input_data[:test_predictions.shape[0]], mask[:test_predictions.shape[0]]*test_predictions))
        print('Final test rmse:',rmse)
        
        # np.save('predictions.npy',test_predictions)


training_data, testing_data = form_user_item_matrix()
print('Done loading data')
sess = tf.InteractiveSession()

no_users, no_items = training_data.shape
autoencoder = Autoencoder(no_items)
autoencoder.train(sess, training_data)
autoencoder.test(sess,testing_data)
            

