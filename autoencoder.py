import tensorflow as tf
import numpy as np

def get_weight_variable(name,shape):
	initializer = tf.layers.xavier_initializer()
	weight = tf.get_variable(name=name, shape=shape, initializer=initializer)
	return weight

def get_bias_variable(name,shape):
	bias = tf.Variable(tf.constant(0), shape=shape,name=name)
	return bias

class Autoencoder():
	def __init__(self, parameters, visible_dim):
		self.visible_dim = visible_dim
		self.parameters = parameters
		self.learningRate = 0.001
		self.epochs = 100
		self.batchsize = 30
		
		self.W1_enc = get_weight_variable('encoder_W1', [visible_dim, 128])
		self.b1_enc = get_bias_variable('encoder_b1',[128])
		self.W2_enc = get_weight_variable('encoder_W2', [128, 128])
		self.b2_enc = get_bias_variable('encoder_b2',[128])
		
		self.W1_dec = get_weight_variable('decoder_W1', [128, 128])
		self.b1_dec = get_bias_variable('decoder_b1',[128])
		self.W2_dec = get_weight_variable('decoder_W2', [128, visible_dim])
		self.b2_dec = get_bias_variable('decoder_b2',[visible_dim])

		def encoder(self, input):
			h1 = tf.matmul(input,self.W1_enc) + self.b1_enc
			h1 = tf.nn.selu(h1)

			h2 = tf.matmul(h1,self.W2_enc) + self.b2_enc
			h2 = tf.nn.selu(h2)

			h3 = tf.nn.dropout(h2,0.8)
			return h3

		def decoder(self, encoder_out):
			h1 = tf.matmul(encoder_out, self.W1_dec) + self.b1_dec
			h1 = tf.nn.selu(h1)

			h2 = tf.matmul(h1, self.W2_dec) + self.b2_dec
			h2 = tf.nn.selu(h2)

			return h2

		def createGraph(self):
			self.X = tf.placeholder(tf.float32, [None, self.visible_dim])

			encoder_out = self.encoder(self.X)
			decoder_out = self.decoder(encoder_out)
			
			mask = #Compute mask 
			#TODO: Adjust the loss term according to the rated and unrated
			loss = tf.losses.mean_squared_error(self.X, mask*decoder_out)
			optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(loss)
			return decoder_out, mask*decoder_out, optimizer

		def train(self, sess, input_data):
			init = tf.global_variables_initializer()
        	sess.run(init)
        	final_predictions, train_predictions, optimize = self.createGraph()
        	self.saver = tf.train.Saver()
	        for epoch in range(self.epochs):
	            results = []
	            test_preds = []
	            for start, end in zip(range(0,len(input_data),self.batchsize), range(self.batchsize,len(input_data)+1,self.batchsize)):
	                predictions, train_preds, _ = sess.run([final_predictions, train_predictions, optimize], feed_dict={self.X: input_data[start:end]})
	                
	                results.append(train_preds)
	                test_preds.append(predictions)
	              
	            print('End of epoch:', epoch)        
	            
	            # if epoch%7 == 0:
	            #     self.saver.save(sess, os.path.join(self.model_path, 'model'), global_step=epoch)
	                
	            total_train_predictions = np.concatenate(results,axis=0)
	            rmse = math.sqrt((unrated/rated+1)*mean_squared_error(input_data, total_train_predictions))
	            print('Final rmse:',rmse)
	        test_preds = np.concatenate(test_preds,axis=0)
	        # self.saver.save(sess, os.path.join(self.model_path, 'model'), global_step=epoch)
	        # np.save('predictions.npy',test_preds)
	        

