import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Read in datasets
movies_df = pd.read_csv('/Users/IXChris/projects/tensorflow_intro/resources/data/ml-1m/movies.dat', sep='::', header=None)
ratings_df = pd.read_csv('/Users/IXChris/projects/tensorflow_intro/resources/data/ml-1m/ratings.dat', sep='::', header=None)

movies_df.columns = ['movie_id', 'title', 'genres']
ratings_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

movies_df['list_index'] = movies_df.index

# Merge data frames and drop unnessary columns
merged_df = movies_df.merge(ratings_df, on='movie_id')
merged_df = (merged_df.
             drop('timestamp', axis=1).
             drop('title', axis=1).
             drop('genres', axis=1)
             )
print(merged_df.head())
# Group by user id
user_group = merged_df.groupby('user_id')

# Amount of users used for training
amount_used_users = 1000

# Create training list
train_X = []

# For each user in group
for user_id, cur_user in user_group:
    # Create temp that stores every movie's rating
    temp = [0] * len(movies_df)
    # For each movie im cur_user's movie list
    for num, movie in cur_user.iterrows():
        # Divide the ratings by 5 and store
        temp[movie['list_index']] = movie['rating'] / 5.0
    # Add the list of ratings into the training list
    train_X.append(temp)

# Build RBM with tf
hidden_units = 20
visible_units = len(movies_df)

# Number of unique movies
vb = tf.placeholder('float', [visible_units])

# Number of features to learn
hb = tf.placeholder('float', [hidden_units])

# Set weights
W = tf.placeholder('float', [visible_units, hidden_units])

# Phase 1: Input Processing
v0 = tf.placeholder('float', [None, visible_units])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))

# Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

# -- Set learning rate and create gradients, CD, & biases ---

# Learning rate
alpha = 1.0

# Gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

# Calculate contrastive divergence
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Create methods to update weights & biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set error function
err = v0 - v1
err_sum = tf.reduce_mean(err * err)

# --- Initialize variables ---
# Current Weight
cur_w = np.zeros([visible_units, hidden_units], np.float32)
# Crrent visible unit biases
cur_vb = np.zeros([visible_units], np.float32)
# Current hidden unit biases
cur_hb = np.zeros([hidden_units], np.float32)
# Previous weights
prv_w = np.zeros([visible_units, hidden_units], np.float32)
# Previous visible unit biases
prv_vb = np.zeros([visible_units], np.float32)
# Previous hidden unit biases
prv_hb = np.zeros([hidden_units], np.float32)
# Set session and run init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# --- Training loop ---

epochs = 15
batchsize = 100
errors = []

for i in range(epochs):
    for start, end in zip(range(0, len(train_X), batchsize),
                          range(batchsize, len(train_X), batchsize)):
        # define batch and feed_dict for update sequence
        batch = train_X[start:end]
        fdict = {v0: batch,
                 W: prv_w,
                 vb: prv_vb,
                 hb: prv_hb}
        # Calculate current values and update weighs & biases
        cur_w = sess.run(update_w, fdict)
        cur_vb = sess.run(update_vb, fdict)
        cur_hb = sess.run(update_hb, fdict)
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb

    # Calculate and record errors for this loop
    errors.append(sess.run(err_sum, feed_dict={v0: train_X,
                                               W: cur_w,
                                               vb: cur_vb,
                                               hb: cur_hb}))
# Plot errors by epoch
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()

# --- Recomendation ---

# Selecting the input user
input_user = [train_X[75]]

# Feed in the user and reconstruct the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: input_user,
                                W: prv_w,
                                hb: prv_hb})

rec = sess.run(vv1, feed_dict={hh0: feed,
                               W: prv_w,
                               vb: prv_vb})

movies_df['recommendation_score'] = rec[0]
print(movies_df.sort(['recommendation_score'], ascending=False).head(20))
