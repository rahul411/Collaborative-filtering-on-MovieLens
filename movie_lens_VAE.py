import itertools
import os

import numpy as np
import tensorflow as tf

from vae_model import VAECF

# 1M dataset

num_user = 6040
num_item = 3952


hidden_encoder_dim = 300
hidden_decoder_dim = 300
latent_dim = 50
# output_dim = 50
learning_rate = 0.002
batch_size = 64
reg_param = 0.02
beta = 0.2
k = 20

n_steps = 1000

one_hot = False


def read_dataset():
    data = np.zeros([num_user, num_item])
    with open('./data/ml-1m/ratings.dat', 'r') as f:
        for line in f.readlines():
            tokens = line.split("::")
            user_id = int(tokens[0]) - 1  # 0 base index
            item_id = int(tokens[1]) - 1
            rating = int(tokens[2])
            data[user_id, item_id] = rating
    return data


def train():
    data = read_dataset()

    num_rating = np.count_nonzero(data)
    idx = np.arange(num_rating)
    np.random.seed(0)
    np.random.shuffle(idx)

    train_idx = idx[:int(0.8 * num_rating)]
    valid_idx = idx[int(0.8 * num_rating):int(0.9 * num_rating)]
    test_idx = idx[int(0.9 * num_rating):]

    result_path = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
        hidden_encoder_dim, hidden_decoder_dim, latent_dim, learning_rate, batch_size, reg_param, one_hot)
    if not os.path.exists(result_path + "/model.ckpt.index"):
    	with tf.Session() as sess:
    		model = VAECF(sess, num_user, num_item,
                          hidden_encoder_dim=hidden_encoder_dim, hidden_decoder_dim=hidden_decoder_dim,
                          latent_dim=latent_dim, learning_rate=learning_rate, 
                          batch_size=batch_size, reg_param=reg_param, one_hot=one_hot, k=k, beta = beta)

    		print("Train size={0}, Validation size={1}, Test size={2}".format(
                train_idx.size, valid_idx.size, test_idx.size))

    		best_mse, best_recall = model.train_test_validation(
                data, train_idx=train_idx, test_idx=test_idx, valid_idx=valid_idx, n_steps=n_steps, result_path=result_path)


    		print("Best MSE = {0}, best Recall = {1}".format(
                best_mse, best_recall))


if __name__ == '__main__':
    train()
