# -*- coding: utf-8 -*-
'''
CVAE MoG (Mixture of Gaussians)
'''
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda, concatenate
from keras.layers import merge, RepeatVector, ConvLSTM2D, Conv3D, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import scipy.io as scio
import gzip
from six.moves import cPickle
import sys
import  theano
import  theano.tensor as T
import math
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
from tqdm import tqdm
from utils_ import *
from itertools import combinations
import math
import tensorflow as tf
import warnings
import keras
import cv2
import sys
from classification_models.keras import Classifiers
from keras import applications

warnings.filterwarnings("ignore")
print(keras.__version__)
print(tf.__version__)

def floatX(X):
    return np.asarray(X, dtype="float32")

#DPP loss
def compute_diversity_loss(h_fake, h_real):
    def compute_diversity(h):
        h = tf.nn.l2_normalize(h, 1)
        #h = K.l2_normalize(h, 1)

        Ly = tf.tensordot(h, tf.transpose(h), 1)
        #Ly = K.dot(h, K.transpose(h))
        eig_val, eig_vec = tf.self_adjoint_eig(Ly)
        #eig_val, eig_vec = K.theano.tensor.nlinalg.eig(Ly)
        return eig_val, eig_vec

    def normalize_min_max(eig_val):
        return tf.div(tf.subtract(eig_val, tf.reduce_min(eig_val)),
                      tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val)))  # Min-max-Normalize Eig-Values

        #return (eig_val - K.min(eig_val)) / (K.max(eig_val) - K.min(eig_val))

    fake_eig_val, fake_eig_vec = compute_diversity(h_fake)
    real_eig_val, real_eig_vec = compute_diversity(h_real)
    # Used a weighing factor to make the two losses operating in comparable ranges.
    eigen_values_loss = 0.0001 * tf.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
    #eigen_values_loss = 0.0001 * K.mean(K.square(real_eig_val - fake_eig_val))
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)

    #eigen_vectors_loss = - K.sum([fake_eig_vec*real_eig_vec])
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    #weighted_eigen_vectors_loss = K.sum(normalized_real_eig_val * eigen_vectors_loss)
    return eigen_values_loss + weighted_eigen_vectors_loss

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
#=====================================

#linear cyclical scheduler for the CVAE MoG loss
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


def gmmpara_init():
    
    theta_init = np.ones(n_centroid)/n_centroid
    u_init = np.zeros((latent_dim, n_centroid))
    lambda_init = np.ones((latent_dim, n_centroid))
    
    #theta_p=theano.shared(np.asarray(theta_init,dtype=theano.config.floatX),name="pi")
    theta_p = tf.Variable(np.asarray(theta_init), dtype="float32")
    #u_p=theano.shared(np.asarray(u_init,dtype=theano.config.floatX),name="u")
    u_p = tf.Variable(np.asarray(u_init), dtype="float32")
    #lambda_p=theano.shared(np.asarray(lambda_init,dtype=theano.config.floatX),name="lambda")
    lambda_p = tf.Variable(np.asarray(lambda_init), dtype="float32")
    return theta_p, u_p, lambda_p

#================================
def get_gamma(tempz):
    #temp_Z=T.transpose(K.repeat(tempz,n_centroid),[0,2,1])
    temp_Z = tf.transpose(K.repeat(tempz, n_centroid), [0,2,1])
    #temp_u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_u_tensor3 = K.repeat_elements(tf.expand_dims(tf.transpose(u_p, [0, 1]), 0), batch_size, axis=0)
    #temp_lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_lambda_tensor3 = K.repeat_elements(tf.expand_dims(tf.transpose(lambda_p, perm=[0, 1]), 0), batch_size, axis=0)
    #temp_theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))
    theta_p1 = tf.expand_dims(theta_p, 0)
    theta_p2 = tf.expand_dims(theta_p1, 0)
    temp_theta_tensor3 = theta_p2 * tf.ones((batch_size,latent_dim,n_centroid))
    temp_p_c_z=K.exp(K.sum((K.log(temp_theta_tensor3)-0.5*K.log(2*math.pi*temp_lambda_tensor3)-\
                       K.square(temp_Z-temp_u_tensor3)/(2*temp_lambda_tensor3)),axis=1))+1e-10
    return temp_p_c_z/K.sum(temp_p_c_z,axis=-1,keepdims=True)
#=====================================================

def loss_cvae(weight):
    def cvae_loss(x, x_decoded_mean):
        reconstruction_loss = K.sum(K.abs(x_decoded_mean - x))
        #reconstruction_loss = objectives.mae(x_decoded_mean, x)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        return reconstruction_loss + weight*kl_loss
    return cvae_loss

def loss(weight):
    def vade_loss(x, x_decoded_mean):
        #Z=T.transpose(K.repeat(z,n_centroid),[0,2,1])
        Z = tf.transpose(K.repeat(z, n_centroid), [0, 2, 1])
        #z_mean_t=T.transpose(K.repeat(z_mean,n_centroid),[0,2,1])
        z_mean_t = tf.transpose(K.repeat(z_mean, n_centroid), [0, 2, 1])
        #z_log_var_t=T.transpose(K.repeat(z_log_var,n_centroid),[0,2,1])
        z_log_var_t = tf.transpose(K.repeat(z_log_var, n_centroid), [0, 2, 1])
        #u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size*train_samples,axis=0)
        #u_tensor3 = K.repeat_elements(u_p.dimshuffle('x', 0, 1), batch_size, axis=0)
        u_tensor3 = K.repeat_elements(tf.expand_dims(tf.transpose(u_p, [0, 1]), 0), batch_size, axis=0)
        #lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size, axis=0)
        lambda_tensor3 = K.repeat_elements(tf.expand_dims(tf.transpose(lambda_p, perm=[0, 1]), 0), batch_size, axis=0)
        #theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size, latent_dim, n_centroid))
        theta_p1 = tf.expand_dims(theta_p, 0)
        theta_p2 = tf.expand_dims(theta_p1, 0)
        theta_tensor3 = theta_p2 * tf.ones((batch_size, latent_dim, n_centroid))

        p_c_z=K.exp(K.sum((K.log(theta_tensor3)-0.5*K.log(2*math.pi*lambda_tensor3)-\
                           K.square(Z-u_tensor3)/(2*lambda_tensor3)),axis=1))+1e-10

        gamma=p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)
        gamma_t=K.repeat(gamma,latent_dim)

        recon_loss = K.sum(K.abs(x_decoded_mean - x))

        #Best of many loss
        #x = K.reshape(x, (batch_size, train_samples, predicting_frame_num, 2))
        #x_decoded_mean = K.reshape(x_decoded_mean, (batch_size, train_samples, predicting_frame_num, 2))
        #rdiff = K.mean(K.abs(x_decoded_mean - x), axis=(0, 1))
        #rdiff_min = K.min(rdiff, axis=1)
        #recon_loss = K.mean(rdiff_min)


        #DPP loss
        # sess = tf.InteractiveSession()
        # sess.run(tf.initialize_all_variables())
        # u_p_ = u_p.eval()
        # lambda_p_ = lambda_p.eval()
        # theta_p_ = theta_p.eval()
        #
        # dpp_loss = 0
        # for i in range(n_centroid):
        #     u = u_p_[:, i]
        #     l = lambda_p_[:, i]
        #
        #     z_sample = np.random.multivariate_normal(u, np.diag(l), (batch_size,)).astype(np.float32)
        #
        #     z_cond = tf.concat([tf.convert_to_tensor(z_sample), c_gru], axis=1)
        #     d_hidden_z_cond = decoder_hidden(z_cond)
        #     d_r = decoder_repeat(d_hidden_z_cond)
        #     d_gru = decoder_gru(d_r)
        #     x_decoded_mean = prediction(d_gru)
        #     x_fake = x_encoded(x_decoded_mean)
        #     x_gru_fake = x_encoded_gru(x_fake)
        #     h_fake = tf.concat([x_gru_fake, c_gru], axis=1)
        #     h_real = inputs
        #     dpp_loss += tf.abs(compute_diversity_loss(h_fake, h_real))

        #h_real = inputs
        # h_real = x_gru
        # x_fake = x_encoded(x_decoded_mean)
        # x_gru_fake = x_encoded_gru(x_fake)
        # #h_fake = tf.concat([x_gru_fake, c_gru], axis=1)
        # h_fake = x_gru_fake
        # dpp_loss = compute_diversity_loss(h_fake, h_real)
        #
        # dpp_weight = 100.0
        # dpp_loss = tf.Print(dpp_loss, [dpp_loss], "Inside loss function")

        kl_loss = K.sum(0.5*gamma_t*(latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)+K.exp(z_log_var_t)/lambda_tensor3+K.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
            -0.5*K.sum(z_log_var+1,axis=-1)\
            #-K.sum(K.log(K.repeat_elements(theta_p.dimshuffle('x',0),batch_size,0))*gamma,axis=-1)\
        -K.sum(K.log(K.repeat_elements(tf.expand_dims(theta_p, 0), batch_size, 0)) * gamma, axis=-1) \
        +K.sum(K.log(gamma)*gamma,axis=-1)

        return recon_loss + (weight*kl_loss) #+ dpp_weight*dpp_loss

    return vade_loss
#================================

#===================================

def get_posterior(z,u,l,sita):
    z_m=np.repeat(np.transpose(z),n_centroid,1)
    posterior=np.exp(np.sum((np.log(sita)-0.5*np.log(2*math.pi*l)-\
                       np.square(z_m-u)/(2*l)),axis=0))
    return posterior/np.sum(posterior,axis=-1,keepdims=True)


dataset = 'mnist'
db = sys.argv[1]
if db in ['mnist','reuters10k','har']:
    dataset = db
ispretrain = False
batch_size = 32
latent_dim = 24
gru_hidden = 128
n_hidden = 128
predicting_frame_num = 12
observed_frame_num = 8
intermediate_dim = [500,500,2000]
theano.config.floatX='float32'
keras.backend.set_floatx('float32')
accuracy=[]
#X,Y = load_data(dataset)
n_epoch = 100
weight = K.variable(0.)
KL_weights = frange_cycle_linear(n_epoch)
train_samples = 1

n_centroid = 5
best_val_loss = 1000
rounds_without_improvement = 0
patience = 2
datatype = "linear"
theta_p,u_p,lambda_p = gmmpara_init()
#===================

#CVAE MoG trajectory
# encoder
x = Input(shape=(predicting_frame_num, 2))
imgs = Input(shape=(observed_frame_num, 50, 40, 3))
scene_map = Input(shape=(observed_frame_num, 72, 90, 2))
condition = Input(shape=(observed_frame_num, 2))

ResNet18, preprocess_input = Classifiers.get('resnet18')



base_model = ResNet18(input_shape=(72, 90, 2), weights=None, include_top=False)
base_model1 = ResNet18(input_shape=(50, 40, 3), weights='imagenet', include_top=False)
imgs_encoded = TimeDistributed(base_model)
flat = TimeDistributed(Flatten())
imgs_encoded_gru = GRU(gru_hidden)

imgs_encoded1 = TimeDistributed(base_model1)
flat1 = TimeDistributed(Flatten())
imgs_encoded_gru1 = GRU(gru_hidden)

#future trajectory
x_encoded = TimeDistributed(Dense(n_hidden, activation='tanh'))
x_encoded_gru = GRU(gru_hidden, return_sequences=False)

#past trajectory
condition_encoded = TimeDistributed(Dense(n_hidden, activation='tanh'))
condition_encoded_gru = GRU(gru_hidden, return_sequences=False)

#imgs_encoded = ConvLSTM2D(filters=64, kernel_size=(3, 3) ,padding='same', return_sequences=False)
#imgs_encoded1 = ConvLSTM2D(filters=32, kernel_size=(3, 3) ,padding='same', return_sequences=True)
#imgs_encoded2 = ConvLSTM2D(filters=16, kernel_size=(3, 3) ,padding='same', return_sequences=False)
#imgs_encoded3 = Conv2D(filters=1, kernel_size=(1, 1),activation='relu',  data_format='channels_last')
#max2d = MaxPooling2D(pool_size=(4, 4), padding='same')
#flat = Flatten()

#scene_encoded = ConvLSTM2D(filters=64, kernel_size=(3, 3) ,padding='same', return_sequences=False)
#scene_encoded1 = ConvLSTM2D(filters=32, kernel_size=(3, 3) ,padding='same', return_sequences=True)
#scene_encoded2 = ConvLSTM2D(filters=16, kernel_size=(3, 3) ,padding='same', return_sequences=False)
#scene_encoded3 = Conv2D(filters=1, kernel_size=(1, 1),activation='relu',  data_format='channels_last')
#max2d_scene = MaxPooling2D(pool_size=(4, 4), padding='same')
#flat_scene = Flatten()

x_enc = x_encoded(x)
x_gru = x_encoded_gru(x_enc)
c_enc = condition_encoded(condition)
c_gru = condition_encoded_gru(c_enc)

#scene context
img_enc = imgs_encoded(scene_map)
flat_ = flat(img_enc)
imgs_gru = imgs_encoded_gru(flat_)

#person appearance
img_enc1 = imgs_encoded1(imgs)
flat_1 = flat1(img_enc1)
imgs_gru1 = imgs_encoded_gru1(flat_1)

#simple attention
#attention_probs = Dense(n_hidden, activation='softmax', name='attention_vec')(c_gru)
#attention_mul = merge([c_gru, attention_probs], name='attention_mul', mode='mul')

encoder_obs = Model([condition], c_gru)
encoder_imgs = Model([scene_map], imgs_gru)
encoder_imgs1 = Model([imgs], imgs_gru1)
#encoder_scene_map = Model( [scene_map], flat_scene_)


#inputs = merge([x_gru, c_gru, flat_, flat_scene_], mode="concat", concat_axis=1)
inputs = concatenate([x_gru, c_gru, imgs_gru, imgs_gru1])
z_mean = Dense(latent_dim)(inputs)
z_log_var = Dense(latent_dim)(inputs)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
#z_cond = merge([z, c_gru, flat_, flat_scene_], mode="concat", concat_axis=1)
z_cond = concatenate([z, c_gru, imgs_gru, imgs_gru1])

# P(X|Y, Z) -- decoder
decoder_hidden = Dense(n_hidden, activation='tanh')
decoder_repeat = RepeatVector(predicting_frame_num)
decoder_gru = GRU(gru_hidden, return_sequences=True)
prediction = TimeDistributed(Dense(2))
d_hidden_z_cond = decoder_hidden(z_cond)
d_r = decoder_repeat(d_hidden_z_cond)
d_gru = decoder_gru(d_r)
x_decoded_mean = prediction(d_gru)


# Generator model, generate new data given latent variable z
d_cond = Input(shape=(gru_hidden,))
d_z = Input(shape=(latent_dim,))
d_imgs = Input(shape=(gru_hidden,))
d_imgs1 = Input(shape=(gru_hidden,))
#d_scene_map = Input(shape=(414,))
#d_inputs = merge([d_z, d_cond, d_imgs, d_scene_map], mode="concat", concat_axis=1)
d_inputs = concatenate([d_z, d_cond, d_imgs, d_imgs1])
d_hidden = decoder_hidden(d_inputs)
d_repeat = decoder_repeat(d_hidden)
d_gru = decoder_gru(d_repeat)
d_predict = prediction(d_gru)
decoder = Model([d_z, d_cond, d_imgs, d_imgs1], d_predict)

#========================
Gamma = Lambda(get_gamma, output_shape=(n_centroid,))(z)
sample_output = Model([x, condition, scene_map, imgs], z_mean)
gamma_output = Model([x, condition, scene_map, imgs], Gamma)
#===========================================      
#vade = Model(x, x_decoded_mean)
cvae_mog = Model([x, condition, scene_map, imgs], x_decoded_mean)
cvae_mog_pretrain = Model([x, condition, scene_map, imgs], x_decoded_mean)
#if ispretrain == True:
    #vade = load_pretrain_weights(vade,dataset)
adam_nn= Adam(lr=1e-5)
adam_gmm= Adam(lr=1e-5)
adam_gmm1= Adam(lr=1e-5)
#vade.compile(optimizer=adam_nn, loss = vae_loss)
cvae_mog_pretrain.compile(optimizer=adam_nn, loss=loss_cvae(weight))
cvae_mog.compile(optimizer=adam_gmm1, loss=loss(weight),add_trainable_weights=[theta_p,u_p,lambda_p],add_optimizer=adam_gmm)
cvae_mog.summary()
#epoch_begin=EpochBegin()
#-------------------------------------------------------

########################DATA#####################


data_train = dict(np.load("data_zara2_train.npz"))
y_tr = data_train['obs_traj_rel']


x_tr = data_train['pred_traj_rel']
x_tr_ = data_train['pred_traj']
y_tr_ = data_train['obs_traj']
#v = data_train['obs_vid']
#v_name = data_train['vid2name']
#frame_idx = data_train['obs_frameidx']
#seq_s_end = data_train['seq_start_end']


data_val = dict(np.load("data_zara2_val.npz"))
x_val = data_val['pred_traj_rel']
x_val_ = data_val['pred_traj']
y_val = data_val['obs_traj_rel']
y_val_ = data_val['obs_traj']

data_test = dict(np.load("data_zara2_test.npz"))
y_te = data_test['obs_traj_rel']
y_te_ = data_test['obs_traj']
x_te = data_test['pred_traj']
v = data_test['obs_vid']
v_name = data_test['vid2name']
seq_s_end = data_test['seq_start_end']
#frame_idx = data_test['obs_frameidx']

#obs_frame_idx = data_test['obs_frameidx']

#Person appearance images
imgs_train = np.load('zara2_imgs_train.npy')
imgs_test = np.load('zara2_imgs_test.npy')

#Scene images
scene_map_train = np.load('zara2_scene_map_train.npy')
scene_map_test = np.load('zara2_scene_map_test.npy')

#pts_img = np.array([[436, 207], [554.6688, 249.7488], [548, 269.2512],[293.3312, 213]])
#pts_wrd = np.array([[8.46, 3.59], [12.09, 5.75], [11.94, 6.77],[3.7, 4.02]])
#h, status = cv2.findHomography(pts_img, pts_wrd)

#Load scene maps
# imgs = np.zeros((y_te_.shape[0], observed_frame_num, 72, 90, 2), np.uint8)
# ped = -1
# for i in tqdm(range(seq_s_end.shape[0])):
#     dataset_name = str(v_name.item(0)[v[seq_s_end[i][0]]])
#     H_path = '/media/atanas/New Volume/M3P/Datasets/eth_ucy/' + dataset_name + '/H.txt'
#
#
#     #invert homography matrix for meter to pixel conversion
#     h = np.linalg.inv(np.loadtxt(H_path))
#     for p in range(seq_s_end[i][0], seq_s_end[i][1]):
#         ped += 1
#         scene_map = cv2.imread('/media/atanas/New Volume/M3P/Datasets/eth_ucy/' + dataset_name + '/annotated.png')
#         scene_map = cv2.cvtColor(scene_map, cv2.COLOR_BGR2GRAY)
#         scene_map[scene_map != 0] = 255
#         scene_map_self = cv2.imread('/media/atanas/New Volume/M3P/Datasets/eth_ucy/' + dataset_name + '/annotated.png')
#         scene_map_self = cv2.cvtColor(scene_map_self, cv2.COLOR_BGR2GRAY)
#         scene_map_self[:] = 255
#         for j in range(observed_frame_num):
#             for f in range(seq_s_end[i][0], seq_s_end[i][1]):
#                 pixel_points = cv2.perspectiveTransform(np.array([[[y_te_[f][j][0], y_te_[f][j][1]]]]), h)
#                 x_pixel, y_pixel = np.squeeze(pixel_points)
#                 if f == p:
#                     cv2.circle(scene_map_self, (x_pixel, y_pixel), 5, (0, 0, 0), -1)
#                 else:
#                     cv2.circle(scene_map, (x_pixel, y_pixel), 5, (0, 0, 0), -1)
#             #cv2.imshow('image', scene_map)
#             #cv2.waitKey(0)
#             #cv2.imshow('image', scene_map_self)
#             #cv2.waitKey(0)
#             scene_map = cv2.resize(scene_map, (90, 72))
#             scene_map_self = cv2.resize(scene_map_self, (90, 72))
#             imgs[ped, j, :, :, 0] = scene_map_self
#             imgs[ped, j, :, :, 1] = scene_map
#
# np.save('eth_scene_map_test.npy', imgs)


#Load person appearance images
# imgs = np.zeros((y_te_.shape[0], observed_frame_num ,50, 40, 3), np.uint8)
# for i in tqdm(range(y_te_.shape[0])):
#     H_path = '/media/atanas/New Volume/M3P/Datasets/eth_ucy/' + str(v_name.item(0)[v[i]]) + '/H.txt'
#     #invert homography matrix for meter to pixel conversion
#     h = np.linalg.inv(np.loadtxt(H_path))
#
#     for j in range(observed_frame_num):
#         pixel_points = cv2.perspectiveTransform(np.array([[[y_te_[i][j][0], y_te_[i][j][1]]]]), h)
#         x_pixel, y_pixel = np.squeeze(pixel_points)
#
#
#         img = cv2.imread('/media/atanas/New Volume/M3P/Datasets/eth_ucy/' + str(v_name.item(0)[v[i]]) +'/images/' + str(v_name.item(0)[v[i]]) +'/'+str(frame_idx[i][j])+'.png')
#         # point_pos means where the trajectory points are located.
#         #video2box = {
#             #"seq_eth": lambda xy: make_box(xy, w=20, h=40, point_pos="head"),
#             #"seq_hotel": lambda xy: make_box(xy, w=80, h=50, point_pos="head"),
#             #"crowds_zara01": lambda xy: make_box(xy, w=50, h=60, point_pos="feet"),
#             #"crowds_zara02": lambda xy: make_box(xy, w=50, h=80, point_pos="head"),
#             #"students003": lambda xy: make_box(xy, w=50, h=80, point_pos="feet")
#         #}
#         if str(v_name.item(0)[v[i]]) == 'crowds_zara01':
#             coords = [int(y_pixel-60), int(y_pixel), int(x_pixel-25), int(x_pixel+25)]
#             coords = [0 if i < 0 else i for i in coords]
#             roi = img[coords[0]:coords[1], coords[2]:coords[3]]
#             roi = cv2.resize(roi, (40, 50))
#             #img_rois.append(roi)
#             #cv2.circle(img, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
#             #cv2.rectangle(img, (int(x_pixel - 25), int(y_pixel - 60)), (int(x_pixel + 25), int(y_pixel)), (255, 0, 0), 2)
#             #cv2.imshow('image', roi)
#             #cv2.waitKey(0)
#         if str(v_name.item(0)[v[i]]) == 'crowds_zara02':
#             coords = [int(y_pixel), int(y_pixel+60), int(x_pixel - 25), int(x_pixel + 25)]
#             coords = [0 if i < 0 else i for i in coords]
#             roi = img[coords[0]:coords[1], coords[2]:coords[3]]
#             roi = cv2.resize(roi, (40, 50))
#             #cv2.circle(img, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
#             #cv2.rectangle(img, (int(x_pixel - 25), int(y_pixel)), (int(x_pixel + 25), int(y_pixel + 60)), (255, 0, 0), 2)
#             #cv2.imshow('image', roi)
#             #cv2.waitKey(0)
#             #img_rois.append(roi)
#
#         if str(v_name.item(0)[v[i]]) == 'students003':
#             coords = [int(y_pixel-60), int(y_pixel), int(x_pixel - 25), int(x_pixel + 25)]
#             coords = [0 if i < 0 else i for i in coords]
#             roi = img[coords[0]:coords[1], coords[2]:coords[3]]
#             roi = cv2.resize(roi, (40, 50))
#             #cv2.circle(img, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
#             #cv2.rectangle(img, (int(x_pixel - 25), int(y_pixel - 60)), (int(x_pixel + 25), int(y_pixel)), (255, 0, 0), 2)
#             #cv2.imshow('image', roi)
#             #cv2.waitKey(0)
#             #img_rois.append(roi)
#
#         if str(v_name.item(0)[v[i]]) == 'seq_eth':
#             coords = [int(y_pixel), int(y_pixel + 40), int(x_pixel - 10), int(x_pixel + 10)]
#             coords = [0 if i < 0 else i for i in coords]
#             roi = img[coords[0]:coords[1], coords[2]:coords[3]]
#             roi = cv2.resize(roi, (40, 50))
#             #cv2.circle(img, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
#             #cv2.rectangle(img, (int(x_pixel - 10), int(y_pixel)), (int(x_pixel + 10), int(y_pixel + 40)), (255, 0, 0), 2)
#             #cv2.imshow('image', roi)
#             #cv2.waitKey(0)
#             #img_rois.append(roi)
#
#         if str(v_name.item(0)[v[i]]) == 'seq_hotel':
#             coords = [int(y_pixel-10), int(y_pixel + 40), int(x_pixel - 20), int(x_pixel + 20)]
#             coords = [0 if i < 0 else i for i in coords]
#             roi = img[coords[0]:coords[1], coords[2]:coords[3]]
#             roi = cv2.resize(roi, (40, 50))
#             #cv2.circle(img, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
#             #cv2.rectangle(img, (int(x_pixel - 20), int(y_pixel - 10)), (int(x_pixel + 20), int(y_pixel + 40)), (255, 0, 0), 2)
#             #cv2.imshow('image', roi)
#             #cv2.waitKey(0)
#
#             #cv2.imshow('image', imgs[i, j, :, :, :])
#             #cv2.waitKey(0)
#         imgs[i, j, :, :, :] = roi
#
# np.save('zara2_imgs_test.npy', imgs)


batch_count = int(x_tr.shape[0] / batch_size)

#######################################################3

#############################Train CVAE with KL annealing and early stopping####################################
# cvae_mog_pretrain.load_weights('CVAE_actev_best.h5')
# for e in range(n_epoch):
#     print("Epoch: ", e)
#     K.set_value(weight, KL_weights[e])
#     #K.set_value(weight, 1.0)
#     for _ in tqdm(range(int(batch_count))):
#         batch_id = np.random.randint(0, x_tr.shape[0], size=batch_size)
#         x_tr_batch = x_tr[batch_id]
#         y_tr_batch = y_tr[batch_id]
#         cvae_mog_pretrain.train_on_batch([x_tr_batch, y_tr_batch], x_tr_batch)
#
#     # val prediction
#
#     z_sample = K.random_normal(shape=(x_val.shape[0], latent_dim))
#     z_sample = K.eval(z_sample)
#
#     obs_traj = y_val
#     enc_condition = encoder.predict([obs_traj])
#     preds = decoder.predict([z_sample, enc_condition])
#
#     x_t = y_val_[:, observed_frame_num - 1, :]
#     x_t = np.expand_dims(x_t, axis=1)
#     x_t = np.repeat(x_t, predicting_frame_num, axis=1)
#     preds = np.cumsum(preds, axis=1)
#     preds = preds + x_t
#
#     preds = np.reshape(preds, [x_val.shape[0], 1, predicting_frame_num, 2])
#
#     current_loss = calc_ade_meters(preds, x_val_)
#     print(current_loss)
#     if current_loss < best_val_loss:
#         best_val_loss = current_loss
#         cvae_mog.save_weights('CVAE_actev_best.h5')
#         rounds_without_improvement = 0
#     else:
#         rounds_without_improvement += 1
#
#     if rounds_without_improvement == patience:
#         print("Early stopping...")
#         break


######################## Test prediction CVAE #############################
# cvae_mog_pretrain.load_weights('CVAE_zara2_best.h5')
# ade = []
# fde = []
# for sample in range(1):
#     z_sample = K.random_normal(shape=(x_te.shape[0], latent_dim))
#     z_sample = K.eval(z_sample)
#
#     obs_traj = y_te
#     enc_condition = encoder.predict([obs_traj])
#     preds = decoder.predict([z_sample, enc_condition])
#
#     x_t = y_te_[:, observed_frame_num - 1, :]
#     x_t = np.expand_dims(x_t, axis=1)
#     x_t = np.repeat(x_t, predicting_frame_num, axis=1)
#     preds = np.cumsum(preds, axis=1)
#     preds = preds + x_t
#     preds = np.reshape(preds, [x_te.shape[0], 1, predicting_frame_num, 2])
#     ade.append(calc_ade_meters(preds, x_te))
#
#     fde.append(calc_fde_meters(preds, x_te))
#
# print("ADE: ", np.min(ade))
# print("FDE: ", np.min(fde))

###############################Pretrain CVAE-MoG#########################3
print("PRETRAINING...")
for e in range(5):
     #don't restrict latent space to unit gaussian (remove KL term)
     K.set_value(weight, 0.0)
     for _ in tqdm(range(int(batch_count))):
         batch_id = np.random.randint(0, x_tr.shape[0], size=batch_size)
         x_tr_batch = x_tr[batch_id]
         y_tr_batch = y_tr[batch_id]
         imgs_batch = imgs_train[batch_id]
         scene_map_batch = scene_map_train[batch_id]
         cvae_mog_pretrain.train_on_batch([x_tr_batch, y_tr_batch, scene_map_batch, imgs_batch], x_tr_batch)

cvae_mog_pretrain.save_weights('CVAE_univ_pretrain.h5')

#Load pretrained CVAE-MoG model and initialize GMM
cvae_mog.load_weights('CVAE_univ_pretrain.h5')
sample = sample_output.predict([x_tr, y_tr, scene_map_train, imgs_train],batch_size=batch_size)
g = mixture.GMM(n_components=n_centroid,covariance_type='diag')
g.fit(sample)
assign_u_p = u_p.assign(floatX(g.means_.T))
assign_lambda_p = lambda_p.assign(floatX(g.covars_.T))
#u_p.set_value(floatX(g.means_.T))
#lambda_p.set_value((floatX(g.covars_.T)))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(assign_u_p)
    sess.run(assign_lambda_p)
    #sess.run(u_p, feed_dict={u_p: floatX(g.means_.T)})
    #sess.run(lambda_p, feed_dict={lambda_p: floatX(g.covars_.T)})
#
#
# ######################################Train CVAE MoG with KL annealing and early stopping######################################3
    print("TRAINING...")
    for e in tqdm(range(1, n_epoch)):
        #K.set_value(weight, 1.0)
        K.set_value(weight, KL_weights[e])
        for _ in range(batch_count):
            batch_id = np.random.randint(0, x_tr.shape[0], size=batch_size)
            x_tr_batch = x_tr[batch_id]
            #x_tr_batch = np.expand_dims(x_tr_batch, axis=1)
            #x_tr_batch = np.repeat(x_tr_batch, train_samples, axis=1)
            #x_tr_batch = np.reshape(x_tr_batch, (batch_size * train_samples, predicting_frame_num, 2))
            y_tr_batch = y_tr[batch_id]
            imgs_train_batch = imgs_train[batch_id]
            scene_map_train_batch = scene_map_train[batch_id]
            #y_tr_batch = np.expand_dims(y_tr_batch, axis=1)
            #y_tr_batch = np.repeat(y_tr_batch, train_samples, axis=1)
            #y_tr_batch= np.reshape(y_tr_batch, (batch_size * train_samples, observed_frame_num, 2))
            cvae_mog.train_on_batch([x_tr_batch, y_tr_batch, scene_map_train_batch, imgs_train_batch], x_tr_batch)
        cvae_mog.save_weights('CVAE_MoG_eth_best.h5')

        # val prediction every 10 epochs

        # if e % 10 == 0:
        #     all_pred = []
        #     u_p_ = []
        #     lambda_p_ = []
        #     theta_p_ = []
        #     u_p_ = u_p.eval()
        #
        #     lambda_p_ = lambda_p.eval()
        #     theta_p_ = theta_p.eval()
        #
        #     for data_point in tqdm(range(int(y_val.shape[0] / batch_size))):
        #         best_p = 0.0
        #         best_pred = []
        #
        #         for j in range(int(20/n_centroid)):
        #             for i in range(n_centroid):
        #
        #                 u = u_p_[:, i]
        #                 l = lambda_p_[:, i]
        #
        #                 z_sample = np.random.multivariate_normal(u, np.diag(l), (batch_size,))
        #
        #                 #p = get_posterior(z_sample, u_p_, lambda_p_, theta_p_)[i]
        #                 #obs_traj = np.expand_dims(y_val[data_point:data_point+batch_size], axis=0)
        #                 obs_traj = y_val[data_point*batch_size:data_point*batch_size+batch_size]
        #                 enc_condition = encoder.predict([obs_traj])
        #                 preds = decoder.predict([z_sample, enc_condition])
        #                 x_t = y_val_[data_point*batch_size:data_point*batch_size+batch_size, observed_frame_num - 1, :]
        #                 x_t = np.expand_dims(x_t, axis=1)
        #                 x_t = np.repeat(x_t, predicting_frame_num, axis=1)
        #                 #x_t = np.expand_dims(x_t, axis=0)
        #                 preds = np.cumsum(preds, axis=1)
        #                 preds = preds + x_t
        #                 #if p > best_p:
        #                     #best_pred = preds
        #                 #all_pred.append(best_pred)
        #                 all_pred.append(preds)
        #
        #     all_pred = np.reshape(all_pred, [int(y_val.shape[0] / batch_size)*batch_size, int(20 / n_centroid)*n_centroid, predicting_frame_num, 2])
        #     current_loss = calc_ade_meters(all_pred, x_val_)
        #     print(current_loss)
        #     if current_loss < best_val_loss:
        #         best_val_loss = current_loss
        #         cvae_mog.save_weights('CVAE_MoG_eth_best.h5')
        #         rounds_without_improvement = 0
        #     else:
        #         rounds_without_improvement += 1
        #
        #     if rounds_without_improvement == patience:
        #         print("Early stopping...")
        #         break


#######################################Test prediction CVAE MoG################################3
    print("TESTING...")
    cvae_mog.load_weights('CVAE_MoG_eth_best.h5')
    all_pred = []
    u_p_ = u_p.eval()
    lambda_p_ = lambda_p.eval()
    #
    lambda_p_ = lambda_p.eval()
    theta_p_ = theta_p.eval()
    for data_point in tqdm(range(x_te.shape[0])):
        best_p = 0.0
        best_pred = []

        for j in range(int(20 / n_centroid)):
            for i in range(n_centroid):
                #u = u_p.eval()[:, i]
                #l = lambda_p.eval()[:, i]
                u = u_p_[:, i]
                l = lambda_p_[:, i]
                z_sample = np.random.multivariate_normal(u, np.diag(l), (1,))
                #p = get_posterior(z_sample, u_p.eval(), lambda_p.eval(), theta_p.eval())[i]
                p = get_posterior(z_sample, u_p_, lambda_p_, theta_p_)[i]
                obs_traj = np.expand_dims(y_te[data_point], axis=0)
                img = np.expand_dims(imgs_test[data_point], axis=0)
                scene_map = np.expand_dims(scene_map_test[data_point], axis = 0)
                enc_condition = encoder_obs.predict([obs_traj])
                enc_img = encoder_imgs.predict([scene_map])
                enc_img1 = encoder_imgs1.predict([img])
                #enc_scene = encoder_scene_map.predict([scene_map])
                preds = decoder.predict([z_sample, enc_condition, enc_img, enc_img1])
                x_t = y_te_[data_point, observed_frame_num - 1, :]
                x_t = np.expand_dims(x_t, axis=0)
                x_t = np.repeat(x_t, predicting_frame_num, axis=0)
                x_t = np.expand_dims(x_t, axis=0)
                preds = np.cumsum(preds, axis=1)
                preds = preds + x_t
                #if p > best_p:
                    #best_pred = preds
                all_pred.append(preds)


    all_pred = np.reshape(all_pred, [x_te.shape[0], int(20 / n_centroid)*n_centroid, predicting_frame_num, 2])
    ade = calc_ade_meters(all_pred, x_te)
    print("ADE: ", ade)
    fde = calc_fde_meters(all_pred, x_te)
    print("FDE: ", fde)

