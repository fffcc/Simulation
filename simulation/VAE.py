import tensorflow as tf
import numpy as np
import input_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
 
class LatentAttention():
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        # self.n_hidden = 500
        self.n_z = 2
        self.batchsize = 100


        self.n_z = 2 # numbers of latent diemensions
        self.n_k = 1 # numbers of kernels/ particles
       
        self.stepsize=tf.placeholder(tf.float32, shape=[])
        self.image_size=2
        self.N=10
        self.epsilon=1e-30
        self.epoch=100

        self.images = tf.placeholder(tf.float32, [self.batchsize, self.image_size*1])
        image_matrix = tf.reshape(self.images,[self.batchsize, self.image_size, 1, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (tf.exp(z_stddev) * samples)
        self.guessed_z=guessed_z
        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, self.image_size*1])


        self.generation_loss = tf.reduce_sum(tf.square(self.images -generated_flat),1) 

        # self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_stddev)- z_stddev/2  - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            input_flat = tf.reshape(input_images,[self.batchsize, 2])# self.batchsize, 7*7*32
            input_flat1 = dense(input_flat, 2, (self.n_z*100), "flat1")

            w_mean = dense(input_flat1, (self.n_z*100), self.n_z, "w_mean")
            w_stddev = dense(input_flat1, (self.n_z*100), self.n_z, "w_stddev")# self.batchsize,self.n_z
            print w_mean
        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        theta=tf.Variable(np.array([[3, -1], [1, -3]]),trainable=False, dtype=tf.float32)
        return tf.matmul(z,theta)

    def train(self):
        z_sample=np.zeros([self.batchsize,self.n_z])
        # visualization = self.mnist.train.next_batch(self.batchsize)[0]
        # reshaped_vis = visualization.reshape(self.batchsize,self.image_size,self.image_size)

        name='./VAE_result/'
        # self.name=name
        res={}
        try:
            os.mkdir(name)
        except:
            None
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)
                    import pdb
                    # pdb.set_trace()
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to printcost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                        # saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test,mean = sess.run((self.generated_images,self.guessed_z), feed_dict={self.images: batch})
                        import pandas as pd
                        import seaborn as sns
                        ssframe=pd.DataFrame(generated_test,columns=['X','Y'])
                        sns.jointplot('X','Y', ssframe, kind = 'kde')
                        plt.savefig(name+'test'+str(epoch)+'.jpg')

                        ssframe=pd.DataFrame(mean[:,:],columns=['X','Y'])
                        sns.jointplot('X','Y', ssframe, kind = 'kde')
                        plt.savefig(name+'mean'+str(epoch)+'.jpg')

                        f = open(name+'test'+str(epoch)+'.txt', "w") 
                        print >> f,mean
                        # generated_test = generated_test.reshape(self.batchsize,28,28)
                        # ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))


model = LatentAttention()
model.train()
