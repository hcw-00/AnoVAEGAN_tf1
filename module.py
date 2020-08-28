from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def encoder(inputs, reuse=False):
    with tf.variable_scope("vae_encoder"):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()): #, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            #inputs = slim.flatten(inputs)
            en_dim = 32
            net = slim.conv2d(inputs=inputs, num_outputs=en_dim, kernel_size=[3, 3], stride=2,normalizer_fn=slim.batch_norm, scope='conv1') #128x128xen_dim
            net = slim.conv2d(inputs=net, num_outputs=en_dim*2, kernel_size=[3, 3], stride=2, normalizer_fn=slim.batch_norm, scope='conv2') #64x64xen_dim*2
            net = slim.conv2d(inputs=net, num_outputs=en_dim*4, kernel_size=[3, 3], stride=2, normalizer_fn=slim.batch_norm, scope='conv3') #32x32xen_dim*4
            net = slim.conv2d(inputs=net, num_outputs=en_dim*4, kernel_size=[3, 3], stride=2, normalizer_fn=slim.batch_norm, scope='conv4') #16x16xen_dim*8
            net = slim.conv2d(inputs=net, num_outputs=en_dim*4, kernel_size=[3, 3], stride=1, normalizer_fn=slim.batch_norm, scope='conv5') #16x16xen_dim*8
            mean = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=1, normalizer_fn=slim.batch_norm, scope='conv6_1') #16x16
            covariance = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=1, normalizer_fn=slim.batch_norm, scope='conv6_2') #16x16

        return mean, covariance

def decoder(inputs, reuse=False):
    with tf.variable_scope("vae_decoder"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()): #, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            de_dim = 32
            net = slim.conv2d_transpose(inputs, de_dim*8, 3, 1, scope="deconv1") # 16x16xde_dim*8
            net = slim.conv2d_transpose(net, de_dim*8, 3, 1, scope="deconv2") # 16x16xde_dim*8
            net = slim.conv2d_transpose(net, de_dim*4, 3, 2, scope="deconv3") # 32x32xde_dim*4
            net = slim.conv2d_transpose(net, de_dim*2, 3, 2, scope="deconv4") # 64x64xde_dim*2
            net = slim.conv2d_transpose(net, de_dim, 3, 2, scope="deconv5") # 128x128xde_dim
            net = slim.conv2d_transpose(net, 1, 3, 2, scope="deconv6") # 256x256x1
        net = slim.conv2d(net, 1, 3, 1, activation_fn=None, scope="deconv7") # 256x256x1
        return tf.nn.tanh(net)

def discriminator(inputs, reuse=False):
    with tf.variable_scope("discriminator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()): #, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            net = slim.conv2d(inputs, 32, 4, 2, scope='conv_0')
            for i in range(3):
                net = slim.conv2d(net, 32, 4, 2, scope='conv_%d' % (i+1))
            net = slim.conv2d(net, 32*6, 4, 2, scope='conv_-2')
            net = slim.conv2d(net, 1, 4, 2, scope='conv_-1')
            net = slim.conv2d(net, 1, 1, 1, scope='conv_-0')
    return net
    
    
    
def feature_extraction_network(inputs, reuse=False):
    '''
    input : 64,64,1
    conv_1 : 64,64,64
    conv_2 : 64,64,64
    pool_1 : 32,32,64
    conv_3 : 32,32,128
    pool_2 : 16,16,128
    conv_4 : 16,16,256
    pool_3 : 8,8,256
    feature layer : 512
    '''
    with tf.variable_scope("feature_extraction_network"):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()): #, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            #inputs = slim.flatten(inputs)
            net = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv1') 
            net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv2') 
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
            net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv3') 
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
            net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv4')
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool3')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 512)
            net = tf.expand_dims(net, axis=2)
        return net


def prediction_network(inputs, reuse=False):

    with tf.variable_scope("prediction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()):
            inputs = slim.flatten(inputs)
            net_a = slim.fully_connected(inputs, 512)
            net_a = slim.fully_connected(net_a, 512)
            spectrum_a = slim.fully_connected(net_a, 101)

            net_b = slim.fully_connected(inputs, 512)
            net_b = slim.fully_connected(net_b, 512)
            spectrum_b = slim.fully_connected(net_b, 101)

            spectra = tf.concat([spectrum_a, spectrum_b], axis=1)
            spectra = tf.expand_dims(spectra, axis=2)
        return spectra


def recognition_network(feature, spectra, latent_dims, reuse=False):

    with tf.variable_scope("recognition_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()):
            spectra = slim.flatten(spectra)
            feature = slim.flatten(feature)
            spectrum_a, spectrum_b = spectra[:,:101], spectra[:,101:]
            net = tf.concat([feature, spectrum_a, spectrum_b], axis=1)
            mean = slim.fully_connected(net, latent_dims)
            covariance = slim.fully_connected(net, latent_dims)

        return mean, covariance

def reconstruction_network(spectra, latent_variables, reuse=False):

    with tf.variable_scope("reconstruction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected], activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()):
            spectra = slim.flatten(spectra)
            spectrum_a, spectrum_b = spectra[:,:101], spectra[:,101:]
            net = tf.concat([spectrum_a, spectrum_b, latent_variables], axis=1)
            net = slim.fully_connected(net, 512) # 512
            net = slim.fully_connected(net, 512) # 512
            net = slim.fully_connected(net, 8*8*256) # 8*8*256
            net = tf.reshape(net, [-1,8,8,256])
            net = slim.conv2d_transpose(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=2) # (16, 16, 128)
            net = slim.conv2d_transpose(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=2) # (32, 32, 64)
            net = slim.conv2d_transpose(inputs=net, num_outputs=1, kernel_size=[3, 3], stride=2)  # (64, 64, 1)

        return net

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop