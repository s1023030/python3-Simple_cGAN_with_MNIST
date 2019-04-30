import tensorflow as tf

import numpy as np

import datetime

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

batch_size=128
z_dimensions = 200

def discriminator(images, reuse_variables=None):
	# Because discriminator is used twice per epoch, so parameter "reuse" should be adopted.
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        d_w1 = tf.get_variable('d_w1', [3, 3, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w2 = tf.get_variable('d_w2', [3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w3 = tf.get_variable('d_w3', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [128], initializer=tf.constant_initializer(0))
        d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 1, 1, 1], padding='SAME')
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)
        d3 = tf.nn.avg_pool(d3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w4 = tf.get_variable('d_w4', [4 * 4 * 128, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1024], initializer=tf.constant_initializer(0))
        d4 = tf.reshape(d3, [-1, 4* 4 * 128])
        d4 = tf.matmul(d4, d_w4)
        d4 = d4 + d_b4
        d4 = tf.nn.relu(d4)

        # Two Branches
		# d5 evaluate how real the image is
		# d6 classfy the number
        d_w5 = tf.get_variable('dv_w1', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b5 = tf.get_variable('dv_b1', [1], initializer=tf.constant_initializer(0))
        d5 = tf.matmul(d4, d_w5) + d_b5
        
        d_w6 = tf.get_variable('dc_w1', [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b6 = tf.get_variable('dc_b1', [1], initializer=tf.constant_initializer(0))
        d6 = tf.matmul(d4, d_w6) + d_b6
    return d5,d6

def generator(z, batch_size, z_dim,c_dim):
    # Combine Z(noise vector) and C(condition of number) and transform to 56*56 dimension
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

# Define the plceholder and the graph
# noise vectors
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# label data : labels correspond to x_placeholder(real data)
c_placeholder=tf.placeholder(tf.float32, [None, 10], name='c_placeholder')
# real data
x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder') 

# generated images
Gz = generator(tf.concat([z_placeholder,c_placeholder],1), batch_size, z_dimensions,10 ) 

# Dx is for real data
# Dx_v evaluate how real the images are
# Dx_c holds the results of number classification 
Dx_v,Dx_c = discriminator(x_placeholder)
# Dg is for fake data
# Dg_v evaluate how real the images are
# Dg_c holds the results of number classification 
Dg_v,Dg_c = discriminator(Gz, reuse_variables=True)

# Loss function for discriminator
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx_v, labels = tf.ones_like(Dx_v)))
d_loss_real_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx_c, labels = c_placeholder))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg_v, labels = tf.zeros_like(Dg_v)))

# Loss function for generator
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg_v, labels = tf.ones_like(Dg_v)))
g_loss_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg_c, labels = c_placeholder))

# Get the varaibles for different network
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]
dv_vars=d_vars.copy()
dv_vars.append([var for var in tvars if 'dv_' in var.name])
dc_vars=d_vars.copy()
dc_vars.append([var for var in tvars if 'dc_' in var.name])

# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.00003).minimize(d_loss_fake, var_list=dv_vars)
d_trainer_real = tf.train.AdamOptimizer(0.00003).minimize(d_loss_real, var_list=dv_vars)
d_trainer_real_c = tf.train.AdamOptimizer(0.00003).minimize(d_loss_real_c, var_list=dc_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.00001).minimize(g_loss, var_list=g_vars)
g_trainer_c = tf.train.AdamOptimizer(0.00001).minimize(g_loss_c, var_list=g_vars)

""" Start Training Session """
saver = tf.train.Saver(max_to_keep=50)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(300):

    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_data_batch=mnist.train.next_batch(batch_size)
    real_image_batch = real_data_batch[0].reshape([batch_size, 28, 28, 1])
    real_label_batch = real_data_batch[1].reshape([batch_size, 10])
    _, _c,__, dLossReal,dLossReal_c,dLossFake = sess.run([d_trainer_real,d_trainer_real_c, d_trainer_fake, d_loss_real,d_loss_real_c ,d_loss_fake],
    feed_dict={x_placeholder: real_image_batch,c_placeholder:real_label_batch, z_placeholder: z_batch})
    if(i % 100 == 0):
        print("dLossReal:", dLossReal,"dLossReal_c:", dLossReal_c, "dLossFake:", dLossFake)
    
# Train generator and discriminator together
for i in range(1000000):

    real_data_batch=mnist.train.next_batch(batch_size)
    real_image_batch = real_data_batch[0].reshape([batch_size, 28, 28, 1])
    real_label_batch = real_data_batch[1].reshape([batch_size, 10])

    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    # Train discriminator on both real and fake images

    _, _c,__, dLossReal,dLossReal_c,dLossFake = sess.run([d_trainer_real_c, d_trainer_fake,d_trainer_real, d_loss_real,d_loss_real_c ,d_loss_fake],
                                            feed_dict={z_placeholder: z_batch,c_placeholder:real_label_batch,x_placeholder:real_image_batch})
    
    # Train generator
    real_data_batch=mnist.train.next_batch(batch_size)
    real_label_batch = real_data_batch[1].reshape([batch_size, 10])
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _,_c = sess.run([g_trainer,g_trainer_c], feed_dict={z_placeholder:z_batch,c_placeholder:real_label_batch})
    
    if i%100==0:
		# Do validation
        real_data_batch=mnist.validation.next_batch(batch_size)
        real_image_batch = real_data_batch[0].reshape([batch_size, 28, 28, 1])
        real_label_batch = real_data_batch[1].reshape([batch_size, 10])
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        dLossReal,dLossReal_c,dLossFake,gLoss = sess.run([d_loss_real,d_loss_real_c ,d_loss_fake,g_loss],
                                            feed_dict={z_placeholder: z_batch,c_placeholder:real_label_batch,x_placeholder:real_image_batch})
        print("iter ",i," :",dLossReal," ",dLossFake," ",dLossReal_c," ",gLoss)
       
    if i%1000==0:
		# save the model
        save_path = saver.save(sess,"model.ckpt")
sess.close()