import tensorflow as tf
import time
import os
import h5py
import numpy as np
import glob
from func import (
    input_setup,
    checkpoint_dir,
    imsave,
    merge
)
class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size,
                 label_size,
                 c_dim):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.c_dim = c_dim
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        
        # all hyperparameters just set as the paper described

        self.biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.c_dim], name='b3'))
        }

        self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, self.c_dim, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, self.c_dim], stddev=1e-3), name='w3')
        }

        self.pred = self.model()
        
        # MSE loss function
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver() # To save checkpoint
        
        

    def model(self):
    	# two relus and a non-activision function layer for averaging and generate outputs
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3'] 
        return conv3

    def train(self, config):
        
        nx, ny = input_setup(config)

        data_dir = checkpoint_dir(config)
        
        input_, label_ = 0, 0
        with h5py.File(data_dir, 'r') as hf:
        	input_ = np.array(hf.get('input'))
        	label_ = np.array(hf.get('label'))

        # using adam optimizer for stochastic gradient descent
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.global_variables_initializer().run()
        # for output of training situations
        counter = 0
        time_ = time.time()

        self.load(config.checkpoint_dir)
        # Train
        if config.mode:

            print("Training.")
            for ep in range(config.epoch):
                single_batch = len(input_) // config.batch_size
                for idx in range(0, single_batch):
                    batch_images = input_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                    if counter % 50 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-time_, err))
                    if counter % 1000 == 0:
                        self.save(config.checkpoint_dir, counter)
        # Test
        else:
            print("Testing")    
            result = self.pred.eval({self.images: input_})           
            image = merge(result, [nx, ny], self.c_dim)
            result = result.squeeze()
            #test
            print(type(image))

            imsave(image, config.result_dir+'/result.png', config)


    def load(self, checkpoint_dir):
    	# load checkpoint
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Check the checkpoint is exist 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
        else:
            print("\n! cannot load checkpoint \n\n")
    def save(self, checkpoint_dir, step):
    	# save the checkpoint
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)