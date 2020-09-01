from __future__ import division
import os
#from google_drive_downloader import GoogleDriveDownloader as gdd
import time
from glob import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.utils import shuffle

from module import *
from utils import *
import utils

import cv2

class vae(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        #self.image_size = args.fine_size
        self.L1_lambda = args.L1_lambda

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.mse = mse_criterion
        self.checkpoint_dir = args.pj_dir + 'checkpoint'
        self.logs_dir = args.pj_dir + 'logs'
        self.sample_dir = args.pj_dir + 'sample'
        self.test_dir =  args.pj_dir + 'test'
        self.dataset_dir = args.pj_dir + 'data'
        dir_path = "D:/Dataset/mvtec_anomaly_detection/transistor/train/good/"
        #dir_path = "D:/Dataset/mvtec_anomaly_detection/transistor/train/good/"
        self.img_path = self.load_data(dir_path)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self._build_model(args)
        
        self.saver = tf.train.Saver(max_to_keep=100)
        

    def load_data(self, dir_path):
        img_dir_path = dir_path
        img_name = os.listdir(img_dir_path)
        img_path = [img_dir_path + i for i in img_name]
        return img_path

    def _load_batch(self, img_path, idx):
        load_size = 286
        crop_size = 256
        img_batch = []
        for i in range(self.batch_size):
            img = cv2.imread(img_path[i+idx*self.batch_size],0)
            img = cv2.resize(img, (load_size,load_size))
            img = get_random_crop(img, crop_size, crop_size)
            img = img/127.5 - 1
            img = np.expand_dims(img, 2)
            img_batch.append(img)

        return img_batch


    def _build_model(self, args):
        # ref : https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
        # log sigma ref : https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/


        self.kl_weight = 1

        self.input_image = tf.placeholder(tf.float32, [None, 256, 256, 1], name='input')

        self.z_mu, self.z_log_sigma, self.z_sigma = self.encoder(self.input_image,  reuse=False)
        self.z_vae = self.z_mu + tf.random_normal(tf.shape(self.z_sigma)) * self.z_sigma
        self.recon_image = self.decoder(self.z_vae)
        
        self.d_real = self.discriminator(self.input_image, reuse=False)
        self.d_fake = self.discriminator(self.recon_image, reuse=True)

        #### for GP ####
        alpha = tf.random_uniform(shape=[args.batch_size, 1], minval=0., maxval=1.)  # eps
        diff = tf.reshape((self.recon_image - self.input_image), [args.batch_size, np.prod(self.input_image.get_shape().as_list()[1:])])
        x_hat = self.input_image + tf.reshape(alpha * diff, [args.batch_size, *self.input_image.get_shape().as_list()[1:]])
        d_hat = self.discriminator(x_hat, reuse=True)


        # Losses
        self.kl_div = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.z_sigma) - tf.log(tf.square(self.z_sigma)) - 1, axis=1))
        self.l_img = tf.reduce_mean(tf.reduce_mean(mse_criterion(self.input_image, self.recon_image, reduction=False), axis=[1,2,3])) # l_recon과 왜 구분?
        #self.l_fts = 
        self.l_recon = tf.reduce_mean(tf.reduce_mean(mae_criterion(self.input_image, self.recon_image, reduction=False)))
        
        self.l_disc = tf.reduce_mean(self.d_fake) - tf.reduce_mean(self.d_real)
        #### for GP ####
        self.scale = 10
        ddx = tf.gradients(d_hat, x_hat)[0]  # gradient
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))  # slopes
        ddx = tf.reduce_mean(tf.square(ddx - 1.0)) * self.scale  # gradient penalty
        self.l_disc = self.l_disc + ddx #
        
        self.l_gen = -tf.reduce_mean(self.d_fake)
        self.l_enc = self.l_recon + self.kl_weight * self.kl_div


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.t_vars = tf.trainable_variables()
            self.dis_vars = [var for var in self.t_vars if 'Discriminator' in var.name]
            self.gen_vars = [var for var in self.t_vars if 'Generator' in var.name]
            self.enc_vars = [var for var in self.t_vars if 'Encoder' in var.name]

            self.optim_dis = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9).minimize(self.l_disc, var_list=self.dis_vars)
            self.optim_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9).minimize(self.l_gen, var_list=self.gen_vars)
            self.optim_vae = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9).minimize(self.l_enc, var_list=self.enc_vars + self.gen_vars)


    def train(self, args):
        
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, args.epoch_step, 0.96, staircase=False)

        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        
        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(self.checkpoint_dir): 
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):

            self.img_path = shuffle(self.img_path)
            
            batch_idxs = len(self.img_path) // self.batch_size
            
            for idx in range(0, batch_idxs):

                input_batch = self._load_batch(self.img_path, idx)

                # Encoder optimization
                fetches = {
                    'reconstructionLoss':self.l_recon,
                    'enc_loss':self.l_enc,
                    'optimizer_e':self.optim_vae,
                    }
                run = self.sess.run(fetches, feed_dict={self.input_image : input_batch})

                # Generator optimization

                _, recon_image_o = self.sess.run([self.optim_gen, self.recon_image], feed_dict={self.input_image : input_batch})

                # Discriminator optimization
                for _ in range(0, 5):
                    _ = self.sess.run(self.optim_dis, feed_dict={self.input_image : input_batch})

            
                if idx%10==0:
                    print(f'reconstruction loss:{run["reconstructionLoss"]:.8f}')

                    #print(("Epoch: [%2d] [%4d/%4d] time: %4.4f vae loss: %4.4f" % (
                    #    epoch, idx, batch_idxs, time.time() - start_time, vae_loss)))

                    #print(("KL_div_loss : [%4.4f], recon_loss : [%4.4f], gen adv loss : [%4.4f]") %(KL_div_loss, reconstruction_loss, gen_adv_loss))
                
                if idx%(batch_idxs//3) == 0: # save sample image
                    cv2.imwrite(self.sample_dir + '/epoch_'+str(epoch)+'_'+str(idx)+'_recon.bmp',((recon_image_o[0,:,:,0])+1)*127.5)
                    #cv2.imwrite(self.sample_dir + '/epoch_'+str(epoch)+'_input.bmp',((input_img[0,:,:,0])+1)*127.5)

            print("Epoch finish")
            #if epoch%500 == 0:
            #    self.save(self.checkpoint_dir, counter)



    def save(self, checkpoint_dir, step):
        model_name = "dnn.model"
        model_dir = "%s" % (self.dataset_dir)
        #checkpoint_dir = checkpoint_dir + '/' + model_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        checkpoint_dir+'/'+model_name,
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            ckpt_paths = ckpt.all_model_checkpoint_paths
            print(ckpt_paths)
            ckpt_name = os.path.basename(ckpt_paths[-1])
            #temp_ckpt = 'dnn.model-80520'
            #ckpt_name = os.path.basename(temp_ckpt)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0

        batch_idxs = len(self.ds) // self.batch_size

        ds_1 = self.ds
        
        loss_list = []

        df_param_target_all = pd.DataFrame()
        df_param_pred_all = pd.DataFrame()

        for idx in range(0, batch_idxs):

            input_batch, target_batch, _ = self._load_batch(ds_1, idx)

            geo_pred, pred, loss, loss_r = self.sess.run([self.geo_reconstructed_l, self.spectra_l_predicted, self.total_loss, self.loss_r],
                                                feed_dict={self.geo_labeled: input_batch, self.spectrum_target: target_batch})


            loss_list.append(loss_r)

            counter += 1
            if idx%1==0:
                print(("Step: [%4d/%4d] time: %4.4f" % (
                    idx, batch_idxs, time.time() - start_time)))
                #df_param = pd.DataFrame(np.squeeze(input_batch), columns={'param1','param2','param3','param4','param5'}) 
                df_pred = pd.DataFrame(np.squeeze(pred))
                df_target = pd.DataFrame(np.squeeze(target_batch))
                #df_geo_pred =  np.squeeze(geo_pred)

                #df_param_pred = pd.concat([df_param, df_pred], axis=1, sort=False)
                #df_param_target = pd.concat([df_param, df_target], axis=1, sort=False)
                #df_param_param = pd.concat([df_param, df_geo_pred], axis=1, sort=False)
                
                df_param_target_all = pd.concat([df_param_target_all, df_target], axis=0, sort=False)
                df_param_pred_all = pd.concat([df_param_pred_all, df_pred], axis=0, sort=False)

            #df_param_target_all.to_csv(self.test_dir+'/result_test_target.csv', index=False)
            #df_param_pred_all.to_csv(self.test_dir+'/result_test_prediction.csv', index=False)

        print("mean regression loss : ")
        print("total time")
        print(time.time() - start_time)


    def test_reconstruction(self, args):

        self.batch_size = 1

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0

        batch_idxs = len(self.ds) // self.batch_size

        ds_1 = self.ds
        
        loss_list = []

        for idx in range(0, batch_idxs):

            input_batch, target_batch, filename_list = self._load_batch(ds_1, idx)

            for j in range(5):
                latent_vector = list(np.random.normal(0,3,5))
                print(latent_vector)
                latent_vector = np.expand_dims(latent_vector, 0)
                geo_recon = self.sess.run([self.geo_reconstructed], 
                                            feed_dict={self.latent_vector: latent_vector, self.spectrum_target: target_batch})


                print(self.test_dir+'/reconstruction/')
                geo_recon = np.squeeze(geo_recon)
                cv2.imwrite(self.test_dir+'/'+str(filename_list)+'_'+str(latent_vector)+'.bmp',(geo_recon+1)*128)
            