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
import numpy
from tensorflow.python import debug as tf_debug
from scipy.stats import norm
import pdb
import seaborn as sns
np.set_printoptions(threshold='nan')
class LatentAttention(): 
    def __init__(self,n_k=1,ratio=1):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.name=''
        self.n_samples = self.mnist.train.num_examples
        self.const=tf.Variable(0,dtype=tf.float32)
        self.n_z = 2 # numbers of latent diemensions
        self.n_k = n_k # numbers of kernels/ particles
        self.batchsize = 100
        self.ratio =ratio # Ratio of (KDE iteration+Particle):KDE iteration
        self.stepsize=tf.placeholder(tf.float32, shape=[])
        self.image_size=2
        self.N=10
        self.epsilon=1e-30
        self.epoch=100
        self.echo=False #print out intermediate variable
        self.alpha=tf.Variable([1.0/self.n_k for x in xrange(self.n_k)],trainable=False, dtype=tf.float32)
        self.is_to_update_alpha=True
        self.isKDE=tf.placeholder(tf.bool, shape=[])# True: gkernel, False: particle
        self.iszFromInput=tf.placeholder(tf.bool, shape=[])# True: gkernel, False: particle
        self.Inputz= tf.placeholder(tf.float32, [self.batchsize, self.n_z])
        # when this is_to_update_alpha is off, alpha is set as fix value.
        # self.p_x_theta=tf.Variable(dtype=tf.float32,trainable=False,name='')
        self.images = tf.placeholder(tf.float32, [self.batchsize, self.image_size*1])
        image_matrix = tf.reshape(self.images,[self.batchsize, self.image_size, 1, 1])
        self.w_mean_tensor, self.w_log_var_tensor =self.recognition(image_matrix)
        # self.batchsize, self.n_k, self.n_z

        sampled_z=tf.cond(self.iszFromInput,lambda :self.Inputz, lambda:self.sampling()) 
        # sampled_z=self.sampling()       
        # generated_images is the output, while generation is the encoder.
        self.generated_images = self.generation(sampled_z)
        self.generation_loss=self.generationLoss(tf.reshape(self.generated_images, [self.batchsize, self.image_size*1]))
        # batchsize,1-x
        self.latent_loss=self.latentLoss(self.w_mean_tensor,self.w_log_var_tensor)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        # self.batchsize->scaler
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        if self.is_to_update_alpha:
            self.increment_global_alpha_op = tf.assign(self.alpha, self.update_alpha(self.alpha,self.w_mean_tensor,self.w_log_var_tensor))

        # self.alpha=self.update_alpha(self.alpha,self.w_mean_tensor,self.w_log_var_tensor)
        
        tpmtx=tf.reshape(tf.transpose(self.w_mean_tensor,[1,0,2]),[self.n_k,self.batchsize* self.n_z])
        self.diff= tf.reduce_sum(tf.square(tpmtx[:,tf.newaxis,:]-tpmtx[tf.newaxis,:,:]),-1)
        
    def sampling(self):
        with tf.variable_scope("sampling"):
            # return tf.cond(self.isKDE, lambda: self.sampling_gkernel(self.w_mean_tensor,self.w_log_var_tensor), lambda: self.sampling_particle(self.w_mean_tensor))
            samples_normal = tf.random_normal([self.batchsize,self.n_k,self.n_z],0,3,dtype=tf.float32)
            # if isKDE then add noise otherwise return original tensor.
            sample_tensor= tf.cond(self.isKDE, lambda:self.w_mean_tensor+(tf.exp(self.w_log_var_tensor/2) * samples_normal),
                 lambda: self.w_mean_tensor)

            ret=tf.reduce_sum( sample_tensor*tf.to_float(tf.reshape(
                tf.contrib.distributions.OneHotCategorical(probs=self.alpha).sample([self.batchsize]),\
                [self.batchsize,self.n_k,1])),1)
            print 'Ret',ret
            return ret

    def update_alpha(self, x,w_mean_tensor,w_log_var_tensor):
        with tf.variable_scope("latent_loss_call"):
            # w_mean_tensor=tf.Print(w_mean_tensor,[w_mean_tensor],message="w_mean_tensor: ")
            kernel_generation=tf.map_fn(self.generation,tf.transpose(self.w_mean_tensor,[1,0,2]))
            # self.n_k, self.batchsize, self.image_size, self.image_size,1
            print kernel_generation
            
            kernel_generation=tf.reshape(kernel_generation,[self.n_k,self.batchsize,self.image_size*1])
            # self.n_k,self.batchsize,self.image_size*self.image_size
            x_t_difference=self.images[tf.newaxis,:,:]-kernel_generation
            # x_t_difference=tf.map_fn(lambda x:x-self.images,kernel_generation)
            # self.n_k,self.batchsize,self.image_size*self.image_size
            self.p_x_theta1 = - 0.5\
                               * tf.reduce_sum(tf.square(x_t_difference), 2)\
                               * self.N / self.batchsize\
                               * self.stepsize
            self.p_x_theta1=tf.cast(self.p_x_theta1,tf.float64)
            self.p_x_theta = tf.reduce_mean(tf.exp(self.p_x_theta1-tf.reduce_min(self.p_x_theta1)), 1)
            # self.p_x_theta=tf.clip_by_value(self.p_x_theta,1e-40,1e+40)
            # self.p_x_theta=tf.Print(self.p_x_theta, [self.p_x_theta], message="p_x_theta: ")

            # self.out_p_x_theta=tf.Variable(self.p_x_theta)
            # pdb.set_trace()


            def update_alpha_gkernel(x,tsr):
                theta_matrix=tsr[:,tf.newaxis,...]-tsr[:,:,tf.newaxis,...]
                # mtx=tf.reduce_sum(tf.reduce_sum(tf.square(tsr[:,tf.newaxis,...]-tsr[:,:,tf.newaxis,...]),-1)
                    # *self.alpha,1)
                # batchsize, self.n_k, self.n_k->batchsize, self.n_k
                # q_t_z_i=tf.exp(-0.5*tf.reduce_mean(mtx,0) * (-self.stepsize/ self.batchsize))
                # batchsize,n_k,n_k ->n_k,n_k
                q_t_z_i=- 0.5 * tf.reduce_sum(tf.square(theta_matrix), axis=3)*self.alpha/ self.batchsize * (- self.stepsize)
                q_t_z_i=tf.cast(q_t_z_i,tf.float64)
                h=4
                self.q_t_z_i = tf.reduce_mean(tf.reduce_sum(tf.exp(1/h*tf.clip_by_value(q_t_z_i-tf.reduce_min(q_t_z_i),0,20)),2) , axis=0) 
                p_z_i=-0.5*tf.reduce_sum(tf.square(tsr),2)* (self.stepsize/ self.batchsize)
                p_z_i=tf.cast(p_z_i,tf.float64)
                self.p_z_i=  tf.reduce_mean(tf.exp(tf.clip_by_value(p_z_i-tf.reduce_min(p_z_i),0,20)),0)

                # self.q_t_z_i=tf.Print(self.q_t_z_i, [self.q_t_z_i], message="This is q_t_z_i: ")
                # self.p_z_i=tf.Print(self.p_z_i, [self.p_z_i], message="This is p_z_i: ")
                # batchsize, self.n_k, self.n_z->self.n_k  q
                return   self.q_t_z_i*self.p_z_i*self.p_x_theta
            # self.alpha=tf.Print(self.alpha,[self.alpha],'Alpha:')
            # qwe=tf.pow(self.alpha,( 1 - self.stepsize))
            # qwe=tf.Print(qwe,[qwe],'alpha: ')

            def update_alpha_particle(x, w_mean_tensor):

                return tf.cast(tf.pow(self.alpha,( 1 - self.stepsize)),tf.float64) *self.p_x_theta


            ret=tf.cond(self.isKDE, lambda: update_alpha_gkernel(x, w_mean_tensor), \
                lambda: update_alpha_particle(x,w_mean_tensor))
            self.rets=tf.reduce_sum(ret)
            ret=ret/tf.reduce_sum(ret)
            # self.rets=tf.Print(self.rets,[self.rets],'Sum: ')
            # ret=tf.Print(ret,[ret],'Alpha: ')
            return tf.cast(ret,tf.float32)
    # encoder
    def latentLoss(self,w_mean_tensor,w_log_var_tensor):
        # self.batchsize,self.n_z->self.batchsize
        with tf.variable_scope("latent_loss_call"):
            return tf.cond(self.isKDE, 
              lambda: tf.reduce_sum((0.5 * tf.reduce_sum(tf.square(w_mean_tensor) 
            + tf.exp(w_log_var_tensor)- w_log_var_tensor/2 - 1,2) )*self.alpha,1),
              # lambda: tf.reduce_sum((0.5 * tf.reduce_sum(tf.square(w_mean_tensor),2) \
              #   +tf.log(tf.clip_by_value(self.alpha,self.epsilon,1)) )*self.alpha,1))
              lambda: tf.reduce_sum(0.5 * tf.reduce_sum(tf.square(w_mean_tensor),2)*self.alpha,1))

    def generationLoss(self,x):
        # self.batchsize,self.n_z->self.batchsize
        # x=tf.cast(x,tf.float64)
        # images=self.images,tf.float64
        images=self.images
        with tf.variable_scope("generation_loss_call"):
            kernel_generation1=tf.map_fn(self.generation,tf.transpose(self.w_mean_tensor,[1,0,2]))
            # self.n_k, self.batchsize, self.image_size, self.image_size,1
            ss=tf.matmul(tf.reshape(self.alpha,[1,self.n_k]),tf.reduce_sum(tf.square(kernel_generation1-images) ,2))
            print "ss",ss
            # kernel_generation1=tf.reshape(kernel_generation1,[self.n_k*self.batchsize,self.image_size*1])
            # print "kernel", kernel_generation1
            # xx=tf.map_fn(lambda x:tf.reduce_sum(tf.square(images-x) ,1),kernel_generation1)
            # print "x", xx
            # xx=tf.reshape(xx,[self.n_k,self.batchsize])
            # xx=xx*self.alpha
            # return tf.cast(-tf.reduce_sum(images * tf.log(tf.clip_by_value(x,self.epsilon,1)) +
             # (1-images) * tf.log(tf.clip_by_value(1-x,self.epsilon,1)),1),tf.float32)

             # return -tf.reduce_sum(self.images * tf.log(x) +
             # (1-self.images) * tf.log(1-x),1)
            ret=tf.cond(self.isKDE, lambda: ss, \
                lambda: tf.reduce_sum(tf.square(images-x) ,1))
            return ret

    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            # h1 = lrelu(conv2d(input_images, 1, 16*self.n_k, "d_h1")) # self.image_sizexself.image_sizex1 -> 14x14x16
            # h2 = lrelu(conv2d(h1, 16*self.n_k, 32*self.n_k, "d_h2")) # 14x14x16 -> 7x7x32
            # h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32*self.n_k])# self.batchsize, 7*7*32
            print input_images
            input_flat = tf.reshape(input_images,[self.batchsize, 2])# self.batchsize, 7*7*32

            input_flat1 = dense(input_flat, 2, (self.n_k*self.n_z*100), "flat1")
            # The k-th  kernel has n_z dimension
            w_mean_dense = dense(input_flat1, self.n_k*self.n_z*100, (self.n_k*self.n_z), "w_mean_dense") # self.batchsize, n_k,n_z
            w_stddev_dense = dense(input_flat1, self.n_k*self.n_z*100, (self.n_k*self.n_z), "w_stddev_dense")# self.batchsize, n_k,n_z
            w_mean_tensor=tf.reshape(w_mean_dense,[self.batchsize,self.n_k,self.n_z]) # self.batchsize, n_k,n_z
            w_log_var_tensor=tf.reshape(w_stddev_dense,[self.batchsize,self.n_k,self.n_z])# self.batchsize, n_k,n_z

            return w_mean_tensor, w_log_var_tensor

    # decoder
    def generation(self, z):
        print 'Sampled z:',z
        theta=tf.Variable(np.array([[3, -1], [1, -3]]),trainable=False, dtype=tf.float32)
        return tf.matmul(z,theta)

        # g1_dense = dense(z, self.n_z, (100*self.n_z), "g1_dense")
        # output_dense = dense(g1_dense, (100*self.n_z), (self.n_z), "output_dense")
        # with tf.variable_scope("generation"):
            # z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            # z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            # h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            # h2 = conv_transpose(h1, [self.batchsize, self.image_size, self.image_size, 1], "g_h2")
            # h2 = tf.nn.sigmoid(h2)


        # return output_dense

    def train(self):
        n=int(np.sqrt(self.batchsize))
        # grid_x = norm.ppf(np.linspace(0.001, 0.999, n))
        # grid_y = norm.ppf(np.linspace(0.001, 0.999, n))
        # grid_x = np.linspace(-5,5,n)
        # grid_y = np.linspace(-5,5,n)
        # pdb.set_trace()
        # z_sample=np.random.multivariate_normal([0,0],np.array([[1, 0], [0, 1]]),self.batchsize)
        # z_sample=[[xi,yi] for j, xi in enumerate(grid_y) for i, yi in enumerate(grid_x) ]
        # z_sample=np.matrix(z_sample)

        z_sample=np.zeros([self.batchsize,self.n_z])
        # visualization = self.mnist.train.next_batch(self.batchsize)[0]
        # reshaped_vis = visualization.reshape(self.batchsize,self.image_size,self.image_size)

        name='./result_k='+str(self.n_k)+'_ratio='+str(self.ratio)+'/'
        # self.name=name
        res={}
        try:
            os.mkdir(name)
        except:
            None
        # ims(name+"base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        idx_cnt=0
        with tf.Session() as sess:
            # Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
            # (You don't need to worry about the BUILD dependency if you are using a pip
            #  install of open-source TensorFlow.)
            from tensorflow.python import debug as tf_debug

            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            gamma_init =1e+1
            sess.run(tf.initialize_all_variables())
            
            for epoch in range(self.epoch):
                gamma = gamma_init/ (100 + np.sqrt(self.N / self.batchsize * epoch))
                print gamma
                for idx in range(int(self.n_samples / self.batchsize)):    
                    idx_cnt+=1
                    # if epoch == 0 and idx < 10 :
                        # gamma = gamma_init
                    # else:
                    
                    if idx==0:
                        print self.alpha.eval()
                    if self.is_to_update_alpha==False:  
                        input_flag = False
                    else:
                        # input_flag = (idx_cnt)%self.ratio==-1
                        input_flag = False
                        # print idx_cnt
                        # input_flag=0
                        # input_flag=0
                    # pdb.set_trace()
                    batch = self.mnist.train.next_batch(self.batchsize)
                    
                    if self.is_to_update_alpha: 
                        # gen_loss,lat_loss,_,_,diff,rets,p_z_i, q_t_z_i, p_x_theta= sess.run((self.generation_loss,self.latent_loss,self.optimizer,\
                        #     self.increment_global_alpha_op,self.diff,self.rets,self.p_z_i,self.q_t_z_i,self.p_x_theta),\
                        #  feed_dict={self.images: batch, self.isKDE:input_flag,self.stepsize:gamma,\
                        #  self.iszFromInput:False,self.Inputz:z_sample})
                        # print rets,p_z_i, q_t_z_i, p_x_theta
                        gen_loss,lat_loss,_,_,diff,rets= sess.run((self.generation_loss,self.latent_loss,self.optimizer,\
                            self.increment_global_alpha_op,self.diff,self.rets),\
                         feed_dict={self.images: batch, self.isKDE:input_flag,self.stepsize:gamma,\
                         self.iszFromInput:False,self.Inputz:z_sample})
                         # print rets
                        # if numpy.isnan(gen_loss):
                            # pdb.set_trace()
                    # pdb.set_trace()
                    if idx % (self.n_samples - 3) == 0: 
                        import pandas as pd
                        
                        print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                        # saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        # generated_test,mean,var = sess.run((self.generated_images,self.w_mean_tensor, self.w_log_var_tensor), feed_dict={self.images: visualization, 
                        #     self.isKDE:input_flag,self.stepsize:gamma,self.iszFromInput:False,self.Inputz:z_sample})

                        generated_test,mean,var = sess.run((self.generated_images,self.w_mean_tensor, self.w_log_var_tensor), feed_dict={self.images: batch, 
                            self.isKDE:input_flag,self.stepsize:gamma,self.iszFromInput:False,self.Inputz:z_sample})
                        ssframe=pd.DataFrame(generated_test,columns=['X','Y'])
                        sns.jointplot('X','Y', ssframe, kind = 'kde')
                        plt.savefig(name+'test'+str(epoch)+'.jpg')

                        ssframe=pd.DataFrame(mean[:,0,:],columns=['X','Y'])
                        sns.jointplot('X','Y', ssframe, kind = 'kde')
                        plt.savefig(name+'mean'+str(epoch)+'.jpg')

                        f = open(name+'test'+str(epoch)+'.txt', "w") 
                        print >> f,mean,var
                        f.close()
                        # generated_test = generated_test.reshape(self.batchsize,self.image_size,self.image_size)
                        # alpha=self.alpha.eval()
                        # # diff=self.diff.eval()
                        # # print diff
                        # # print alpha
                        # res[epoch]=[np.mean(gen_loss),np.mean(lat_loss),alpha]
                        # ims(name+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))

                        # if self.n_z!=2:
                        #     continue
                        
                        #         # x_decoded = decoder.predict(z_sample)
                        #         # sess.run((self.generated_images),feed_dict={self.images: batch, self.isKDE:input_flag,self.stepsize:gamma,self.iszFromInput:True,self.Inputzs:z_sample})
                        # x_decoded = sess.run(self.generated_images,feed_dict={self.images: batch, self.isKDE:input_flag,self.stepsize:gamma,self.iszFromInput:True,self.Inputz:z_sample})
                        #         # digit = x_decoded[0].reshape(digit_size, digit_size)
                        # # pdb.set_trace()
                        # for j, xi in enumerate(grid_y):
                        #     for i, yi in enumerate(grid_x):
                        #         digit = x_decoded[i  * n+j].reshape(digit_size, digit_size)
                        #         figure[i * digit_size: (i + 1) * digit_size,
                        #            j * digit_size: (j + 1) * digit_size] = digit
                        
                        # plt.figure(figsize=(10, 10))
                        # plt.imshow(figure, cmap='Greys_r')
                        # output_add=name+'update_alpha_models_{}_{}_{}epochs.png'.format(self.n_k, str(self.isKDE), epoch)
                        # print output_add
                        # plt.savefig(output_add)
            import pandas as pd
            res=pd.DataFrame(res)
            res.to_csv(name+'out.csv') 
            sess.close()

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-k")
parser.add_argument("-r")
args = parser.parse_args()
print args.k,args.r
model = LatentAttention(n_k=int(args.k),ratio=int(args.r))
model.train()

    
