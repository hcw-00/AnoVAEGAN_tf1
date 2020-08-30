from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

# Base model : BiGAN architecture
# code source : https://github.com/YOUSIKI/BiGAN.TensorLayer/blob/celeba/model.py

def encoder(inputs, use_batchnorm=True, reuse=False):
    with tf.variable_scope("vae_encoder"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None, weights_initializer = tf.initializers.random_normal(stddev=0.02)):
            net = inputs
            num_pooling = int(math.log(256,2) - math.log(float(8,2)))
            for i in range(num_pooling):
                filters = int(min(128, 32*(2**i)))
                net = slim.conv2d(net, filters, 5, 2, scope='enc_conv2D_'+str(i))
                if use_batchnorm:net = slim.batch_norm(net, activation_fn=tf.nn.leaky_relu)
                else: net = slim.layer_norm(net, activation_fn=tf.nn.leaky_relu)
            
            en_dim = 32
            net = slim.conv2d(inputs=inputs, num_outputs=en_dim, kernel_size=[5, 5], stride=2, activation_fn=tf.nn.leaky_relu, scope='conv1') #128x128xen_dim
            net = slim.conv2d(inputs=net, num_outputs=en_dim*2, kernel_size=[5, 5], stride=2, scope='conv2') #64x64xen_dim*2
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.leaky_relu, scope = 'batch1') # gamma init?
            net = slim.conv2d(inputs=net, num_outputs=en_dim*4, kernel_size=[5, 5], stride=2, scope='conv3') #32x32xen_dim*4
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.leaky_relu, scope = 'batch2')
            net = slim.conv2d(inputs=net, num_outputs=en_dim*8, kernel_size=[5, 5], stride=2, scope='conv4') #16x16xen_dim*8
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.leaky_relu, scope = 'batch3')
            
            mean = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[5, 5], stride=1, scope='conv6_1') #16x16
            covariance = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[5, 5], stride=1, scope='conv6_2') #16x16

        return mean, covariance

def decoder(inputs, reuse=False):
    with tf.variable_scope("vae_decoder"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None, weights_initializer = tf.initializers.random_normal(stddev=0.02)): #, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            de_dim = 32
            net = slim.conv2d_transpose(inputs, de_dim*8, 5, 1, scope="deconv1") # 16x16xde_dim*8 <==check
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.relu, scope = 'batch1') # <==check
            net = slim.conv2d_transpose(net, de_dim*4, 5, 2, scope="deconv2") # 32x32xde_dim*4
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.relu, scope = 'batch2')
            net = slim.conv2d_transpose(net, de_dim*2, 5, 2, scope="deconv3") # 64x64xde_dim*2
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.relu, scope = 'batch3')
            net = slim.conv2d_transpose(net, de_dim, 5, 2, scope="deconv4") # 128x128xde_dim
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.relu, scope = 'batch4')
            net = slim.conv2d_transpose(net, 1, 5, 2, scope="deconv5") # 256x256x1
            net = slim.conv2d(net, 1, 5, 1, activation_fn=None, scope="deconv6") # 256x256x1 <= check
        return tf.nn.tanh(net)

def discriminator(inputs, reuse=False):
    with tf.variable_scope("discriminator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()): #, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            d_dim = 32
            net = slim.conv2d(inputs, d_dim, 5, 2, scope='conv_0')
            net = slim.conv2d(net, d_dim*2, 5, 2, scope='conv_1')
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.leaky_relu, scope = 'batch1') # <==check
            net = slim.dropout(net,0.8)
            net = slim.conv2d(net, d_dim*4, 5, 2, scope='conv_2')
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.leaky_relu, scope = 'batch2') # <==check
            net = slim.dropout(net,0.8)
            net = slim.conv2d(net, d_dim*8, 5, 2, scope='conv_3')
            net = slim.batch_norm(net, decay=0.9, activation_fn = tf.nn.leaky_relu, scope = 'batch3') # <==check
            net = slim.dropout(net,0.8)
            net = slim.fully_connected(net, 1, activation_fn = tf.identity, scope='fc')

    return net  
    

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


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