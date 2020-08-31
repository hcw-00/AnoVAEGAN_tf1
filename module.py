from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

# Base model : BiGAN architecture
# code source : https://github.com/YOUSIKI/BiGAN.TensorLayer/blob/celeba/model.py
# Basem model : https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI

def encoder(inputs, use_batchnorm=True, reuse=False):
    with tf.variable_scope("Encoder"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None, weights_initializer = tf.initializers.random_normal(stddev=0.02)):
            net = inputs
            # down sampling
            num_pooling = int(math.log(256,2) - math.log(float(8),2)) # 5
            for i in range(num_pooling):
                filters = int(min(128, 32*(2**i)))
                net = slim.conv2d(net, filters, 5, 2, scope='enc_conv2D_'+str(i))
                if use_batchnorm: net = slim.batch_norm(net, activation_fn=tf.nn.leaky_relu)
                else: net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)

            # intermedidate conv
            net = slim.conv2d(net, 64, 1, scope='enc_conv2D_interm')
            net = slim.flatten(net)
            net_mu = slim.fully_connected(net, 128, activation_fn = None)
            net_log_sigma = slim.fully_connected(net, 128, activation_fn = None)

            z_mu = slim.dropout(net_mu, 0.8) # <== check dropout rate !
            z_log_sigma = slim.dropout(net_log_sigma, 0.8) # <== check dropout rate !
            z_sigma = tf.exp(z_log_sigma)

            return z_mu, z_log_sigma, z_sigma
            

def decoder(z_vae, use_batchnorm=False, reuse=False):
    with tf.variable_scope("Generator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None, weights_initializer = tf.initializers.random_normal(stddev=0.02)): #, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            
            r_dim = 8*8*64
            # dense
            net = slim.fully_connected(z_vae, r_dim)
            # dropout
            net = slim.dropout(net, 0.8) # <== check
            # reshape
            net = tf.reshape(net, [-1,8,8,64]) # <== check # match with r_dim
            # intermediate conv reverse
            net = slim.conv2d(net, 64, 1)
            # upsampling
            num_upsampling = int(math.log(256,2) - math.log(float(8),2)) # 5
            if use_batchnorm: net = slim.batch_norm(net, activation_fn=tf.nn.relu)
            else: net = slim.layer_norm(net, activation_fn=tf.nn.relu)
            for i in range(num_upsampling):
                filters = int(max(32, 128 / (2 ** i)))
                net = slim.conv2d_transpose(net, filters, 5, 2, scope='dec_conv2DT_'+str(i))
                if use_batchnorm: net = slim.batch_norm(net, activation_fn=tf.nn.leaky_relu)
                else: net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
            net = slim.conv2d(net, 1, 1, activation_fn=None, scope='dec_conv2D_final')

        return net

def discriminator(inputs, use_batchnorm=False, reuse=False):
    with tf.variable_scope("Discriminator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()): #, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            net = inputs
            num_pooling = int(math.log(256,2) - math.log(float(8),2)) # 5
            for i in range(num_pooling):
                filters = int(min(128, 32*(2**i)))
                net = slim.conv2d(net, filters, 5, 2, scope='dis_conv2D_'+str(i))
                if use_batchnorm: net = slim.batch_norm(net, activation_fn=tf.nn.leaky_relu)
                else: net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)            
            d_out = slim.fully_connected(net, 1, activation_fn = None)

        # "for GP" check

    return d_out  
    

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target, reduction=True): # mae??? not mse??
    if reduction:
        return tf.reduce_mean(tf.abs(in_-target))
    else:
        return tf.abs(in_-target)

def mse_criterion(in_, target, reduction=True):
    if reduction:
        return tf.reduce_mean((in_-target)**2)
    else:
        return (in_-target)**2


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

#def bce_criterion(logits, labels):
#    return tf.reduce_mean(tf.nn.sig)

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop