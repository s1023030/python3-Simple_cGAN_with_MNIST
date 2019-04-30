import tensorflow as tf

import numpy as np

import datetime

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

batch_size=1
number=np.array([[0,0,0,0,1,0,0,0,0,0]])
z_dimensions = 200
RESTORE_FROM="model.ckpt"

def generator(z, batch_size, z_dim,c_dim):
    g_w1 = tf.get_variable('g_w1', [z_dim+c_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    return g4


tf.reset_default_graph() 

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
c_placeholder=tf.placeholder(tf.float32, [None, 10], name='c_placeholder')
Gz = generator(tf.concat([z_placeholder,c_placeholder],1), batch_size, z_dimensions,10 ) 

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,RESTORE_FROM)
    z_batch = np.random.normal(0, 1, [1, z_dimensions])
    generated_image = sess.run(Gz,

                                feed_dict={z_placeholder: z_batch,c_placeholder:number})

    generated_image = generated_image.reshape([28, 28])

    plt.imshow(generated_image,cmap='Greys')
    plt.show()
    plt.savefig("img.png")