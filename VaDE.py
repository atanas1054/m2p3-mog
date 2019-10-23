# -*- coding: utf-8 -*-
'''
VaDE (Variational Deep Embedding:A Generative Approach to Clustering)

Best clustering accuracy: 
MNIST: 94.46% +
Reuters10k: 81.66% +
HHAR: 85.38% +
Reuters_all: 79.38% +

@code author: Zhuxi Jiang
'''
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda
from keras.layers import merge, RepeatVector
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

import warnings
warnings.filterwarnings("ignore")

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
#=====================================
def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w
             
#==================================================
def load_data(dataset):
    path = 'dataset/'+dataset+'/'
    if dataset == 'mnist':
        path = path + 'mnist.pkl.gz'
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')
    
        if sys.version_info < (3,):
            (x_train, y_train), (x_test, y_test) = cPickle.load(f)
        else:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")
    
        f.close()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        X = np.concatenate((x_train,x_test))
        Y = np.concatenate((y_train,y_test))
        
    if dataset == 'reuters10k':
        data=scio.loadmat(path+'reuters10k.mat')
        X = data['X']
        Y = data['Y'].squeeze()
        
    if dataset == 'har':
        data=scio.loadmat(path+'HAR.mat')
        X=data['X']
        X=X.astype('float32')
        Y=data['Y']-1
        X=X[:10200]
        Y=Y[:10200]

    return X,Y


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

def config_init(dataset):
    if dataset == 'mnist':
        return 784,3000,10,0.002,0.002,10,0.9,0.9,1,'sigmoid'
    if dataset == 'reuters10k':
        return 2000,15,4,0.002,0.002,5,0.5,0.5,1,'linear'
    if dataset == 'har':
        return 561,120,6,0.002,0.00002,10,0.9,0.9,5,'linear'
        
def gmmpara_init():
    
    theta_init=np.ones(n_centroid)/n_centroid
    u_init=np.zeros((latent_dim,n_centroid))
    lambda_init=np.ones((latent_dim,n_centroid))
    
    theta_p=theano.shared(np.asarray(theta_init,dtype=theano.config.floatX),name="pi")
    u_p=theano.shared(np.asarray(u_init,dtype=theano.config.floatX),name="u")
    lambda_p=theano.shared(np.asarray(lambda_init,dtype=theano.config.floatX),name="lambda")
    return theta_p,u_p,lambda_p

#================================
def get_gamma(tempz):
    temp_Z=T.transpose(K.repeat(tempz,n_centroid),[0,2,1])
    temp_u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))
    
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
        Z=T.transpose(K.repeat(z,n_centroid),[0,2,1])
        z_mean_t=T.transpose(K.repeat(z_mean,n_centroid),[0,2,1])
        z_log_var_t=T.transpose(K.repeat(z_log_var,n_centroid),[0,2,1])
        u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
        lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
        theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))

        p_c_z=K.exp(K.sum((K.log(theta_tensor3)-0.5*K.log(2*math.pi*lambda_tensor3)-\
                           K.square(Z-u_tensor3)/(2*lambda_tensor3)),axis=1))+1e-10

        gamma=p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)
        gamma_t=K.repeat(gamma,latent_dim)

        # if datatype == 'sigmoid':
        #     loss=alpha*original_dim * objectives.binary_crossentropy(x, x_decoded_mean)\
        #     +K.sum(0.5*gamma_t*(latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)+K.exp(z_log_var_t)/lambda_tensor3+K.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
        #     -0.5*K.sum(z_log_var+1,axis=-1)\
        #     -K.sum(K.log(K.repeat_elements(theta_p.dimshuffle('x',0),batch_size,0))*gamma,axis=-1)\
        #     +K.sum(K.log(gamma)*gamma,axis=-1)
        # else:
        recon_loss = K.sum(K.abs(x_decoded_mean - x))
        kl_loss = K.sum(0.5*gamma_t*(latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)+K.exp(z_log_var_t)/lambda_tensor3+K.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
            -0.5*K.sum(z_log_var+1,axis=-1)\
            -K.sum(K.log(K.repeat_elements(theta_p.dimshuffle('x',0),batch_size,0))*gamma,axis=-1)\
            +K.sum(K.log(gamma)*gamma,axis=-1)

        return recon_loss + (weight*kl_loss)
    return vade_loss
#================================

def load_pretrain_weights(vade,dataset):
    ae = model_from_json(open('pretrain_weights/ae_'+dataset+'.json').read())
    ae.load_weights('pretrain_weights/ae_'+dataset+'_weights.h5')
    vade.layers[1].set_weights(ae.layers[0].get_weights())
    vade.layers[2].set_weights(ae.layers[1].get_weights())
    vade.layers[3].set_weights(ae.layers[2].get_weights())
    vade.layers[4].set_weights(ae.layers[3].get_weights())
    vade.layers[-1].set_weights(ae.layers[-1].get_weights())
    vade.layers[-2].set_weights(ae.layers[-2].get_weights())
    vade.layers[-3].set_weights(ae.layers[-3].get_weights())
    vade.layers[-4].set_weights(ae.layers[-4].get_weights())
    sample = sample_output.predict(X,batch_size=batch_size)
    if dataset == 'mnist':
        g = mixture.GMM(n_components=n_centroid,covariance_type='diag')
        g.fit(sample)
        u_p.set_value(floatX(g.means_.T))
        lambda_p.set_value((floatX(g.covars_.T)))
    if dataset == 'reuters10k':
        k = KMeans(n_clusters=n_centroid)
        k.fit(sample)
        u_p.set_value(floatX(k.cluster_centers_.T))
    if dataset == 'har':
        g = mixture.GMM(n_components=n_centroid,covariance_type='diag',random_state=3)
        g.fit(sample)
        u_p.set_value(floatX(g.means_.T))
        lambda_p.set_value((floatX(g.covars_.T)))
    print ('pretrain weights loaded!')
    return vade
#===================================
def lr_decay():
    if dataset == 'mnist':
        adam_nn.lr.set_value(floatX(max(adam_nn.lr.get_value()*decay_nn,0.0002)))
        adam_gmm.lr.set_value(floatX(max(adam_gmm.lr.get_value()*decay_gmm,0.0002)))
    else:
        adam_nn.lr.set_value(floatX(adam_nn.lr.get_value()*decay_nn))
        adam_gmm.lr.set_value(floatX(adam_gmm.lr.get_value()*decay_gmm))
    print ('lr_nn:%f'%adam_nn.lr.get_value())
    print ('lr_gmm:%f'%adam_gmm.lr.get_value())


def get_posterior(z,u,l,sita):
    z_m=np.repeat(np.transpose(z),n_centroid,1)
    posterior=np.exp(np.sum((np.log(sita)-0.5*np.log(2*math.pi*l)-\
                       np.square(z_m-u)/(2*l)),axis=0))
    return posterior/np.sum(posterior,axis=-1,keepdims=True)

    
def epochBegin(epoch):

    if epoch % decay_n == 0 and epoch!=0:
        lr_decay()


    #p=g.predict(sample)
    #acc_g=cluster_acc(p,Y)
    
    if epoch <1 and ispretrain == False:
        sample = sample_output.predict([x_tr, y_tr], batch_size=batch_size)
        g = mixture.GMM(n_components=n_centroid, covariance_type='diag')
        g.fit(sample)
        u_p.set_value(floatX(g.means_.T))
        lambda_p.set_value((floatX(g.covars_.T)))
        print ('no pretrain,random init!')

    #gamma = gamma_output.predict(X,batch_size=batch_size)
    #output = vade.predict(X, batch_size=batch_size)
    #print(output.shape)
    #acc=cluster_acc(np.argmax(gamma,axis=1),Y)
    #global accuracy
    #accuracy+=[acc[0]]
    #if epoch>0 :
        #print ('acc_p_c_z:%0.8f'%acc[0])
    #if epoch==1 and dataset == 'har' and acc[0]<0.77:
        #print ('=========== HAR dataset:bad init!Please run again! ============')
        #sys.exit(0)
        
class EpochBegin(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        epochBegin(epoch)
#==============================================

dataset = 'mnist'
db = sys.argv[1]
if db in ['mnist','reuters10k','har']:
    dataset = db
ispretrain = False
batch_size = 128
latent_dim = 24
gru_hidden = 128
n_hidden = 128
predicting_frame_num = 12
observed_frame_num = 8
intermediate_dim = [500,500,2000]
theano.config.floatX='float32'
accuracy=[]
X,Y = load_data(dataset)
n_epoch = 200
weight = K.variable(0.)
KL_weights = frange_cycle_linear(n_epoch)

n_centroid = 6
best_val_loss = 1000
rounds_without_improvement = 0
patience = 20
datatype = "linear"
theta_p,u_p,lambda_p = gmmpara_init()
#===================

#MNIST VAE
# x = Input(batch_shape=(batch_size, original_dim))
# h = Dense(intermediate_dim[0], activation='relu')(x)
# h = Dense(intermediate_dim[1], activation='relu')(h)
# h = Dense(intermediate_dim[2], activation='relu')(h)
# z_mean = Dense(latent_dim)(h)
# z_log_var = Dense(latent_dim)(h)
# z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
# h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
# h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
# x_decoded_mean = Dense(original_dim, activation=datatype)(h_decoded)

#CVAE MoG trajectory
# encoder
x = Input(shape=(predicting_frame_num, 2))
condition = Input(shape=(observed_frame_num, 2))

x_encoded = TimeDistributed(Dense(n_hidden, activation='tanh'))(x)
x_encoded = GRU(gru_hidden, return_sequences=False)(x_encoded);

condition_encoded = TimeDistributed(Dense(n_hidden, activation='tanh'))
condition_encoded_gru = GRU(gru_hidden, return_sequences=False)
c_enc = condition_encoded(condition)
c_gru = condition_encoded_gru(c_enc)
encoder = Model([condition], c_gru)

inputs = merge([x_encoded, c_gru], mode="concat", concat_axis=1)
z_mean = Dense(latent_dim)(inputs)
z_log_var = Dense(latent_dim)(inputs)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
z_cond = merge([z, c_gru], mode="concat", concat_axis=1)

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
d_inputs = merge([d_z, d_cond], mode="concat", concat_axis=1)
d_hidden = decoder_hidden(d_inputs)
d_repeat = decoder_repeat(d_hidden)
d_gru = decoder_gru(d_repeat)
d_predict = prediction(d_gru)
decoder = Model([d_z, d_cond], d_predict)

#========================
Gamma = Lambda(get_gamma, output_shape=(n_centroid,))(z)
sample_output = Model([x, condition], z_mean)
gamma_output = Model([x, condition], Gamma)
#===========================================      
#vade = Model(x, x_decoded_mean)
cvae_mog = Model([x, condition], x_decoded_mean)
cvae_mog_pretrain = Model([x, condition], x_decoded_mean)
#if ispretrain == True:
    #vade = load_pretrain_weights(vade,dataset)
adam_nn= Adam(lr=1e-5)
adam_gmm= Adam(lr=1e-4)
adam_gmm1= Adam(lr=1e-4)
#vade.compile(optimizer=adam_nn, loss = vae_loss)
cvae_mog_pretrain.compile(optimizer=adam_nn, loss=loss_cvae(weight))
cvae_mog.compile(optimizer=adam_gmm1, loss=loss(weight),add_trainable_weights=[theta_p,u_p,lambda_p],add_optimizer=adam_gmm)
cvae_mog.summary()
#epoch_begin=EpochBegin()
#-------------------------------------------------------


# data load
#y_tr = np.load('nuscenes_input_train.npy')
#x_tr = np.load('nuscenes_output_train.npy')
#y_te = np.load('nuscenes_input_val.npy')
#x_te = np.load('nuscenes_output_val.npy')

data_train = dict(np.load("data_actev_train.npz"))
y_tr = data_train['obs_traj_rel']

x_tr = data_train['pred_traj_rel']
x_tr_ = data_train['pred_traj']
y_tr_ = data_train['obs_traj']

data_val = dict(np.load("data_actev_val.npz"))
x_val = data_val['pred_traj_rel']
x_val_ = data_val['pred_traj']
y_val = data_val['obs_traj_rel']
y_val_ = data_val['obs_traj']

data_test = dict(np.load("data_actev_test.npz"))
y_te = data_test['obs_traj_rel']
y_te_ = data_test['obs_traj']
x_te = data_test['pred_traj']


#print(np.cumsum(x_tr[1],axis=0) + y_tr_[1, observed_frame_num-1, :])

batch_count = int(x_tr.shape[0] / batch_size)

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
# cvae_mog_pretrain.load_weights('CVAE_actev_best.h5')
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

###############################Pretrain CVAE#########################3

# for e in range(20):
#      #don't restrict latent space to unit gaussian (remove KL term)
#      K.set_value(weight, 0.0)
#      for _ in tqdm(range(int(batch_count))):
#          batch_id = np.random.randint(0, x_tr.shape[0], size=batch_size)
#          x_tr_batch = x_tr[batch_id]
#          y_tr_batch = y_tr[batch_id]
#          cvae_mog_pretrain.train_on_batch([x_tr_batch, y_tr_batch], x_tr_batch)
#
# cvae_mog_pretrain.save_weights('CVAE_actev_pretrain.h5')

#Load pretrained  CVAE model and initialize GMM
cvae_mog.load_weights('CVAE_actev_pretrain.h5')
sample = sample_output.predict([x_tr, y_tr],batch_size=batch_size)
g = mixture.GMM(n_components=n_centroid,covariance_type='diag')
g.fit(sample)
u_p.set_value(floatX(g.means_.T))
lambda_p.set_value((floatX(g.covars_.T)))
#
#
# ######################################Train CVAE MoG with KL annealing and early stopping######################################3
for e in range(n_epoch):
    #K.set_value(weight, 1.0)
    K.set_value(weight, KL_weights[e])
    for _ in tqdm(range(batch_count)):
        batch_id = np.random.randint(0, x_tr.shape[0], size=batch_size)
        x_tr_batch = x_tr[batch_id]
        y_tr_batch = y_tr[batch_id]
        cvae_mog.train_on_batch([x_tr_batch, y_tr_batch], x_tr_batch)

    # val prediction
    all_pred = []
    for data_point in tqdm(range(x_val.shape[0])):
        best_p = 0.0
        best_pred = []

        #for j in range(int(20/n_centroid)):
        for i in range(n_centroid):
            u = u_p.eval()[:, i]
            l = lambda_p.eval()[:, i]
            z_sample = np.random.multivariate_normal(u, np.diag(l), (1,))
            #print("Cluster ", str(i))
            p = get_posterior(z_sample, u_p.eval(), lambda_p.eval(), theta_p.eval())[i]
            #print(p)
            obs_traj = np.expand_dims(y_val[data_point], axis=0)
            enc_condition = encoder.predict([obs_traj])
            preds = decoder.predict([z_sample, enc_condition])
            x_t = y_val_[data_point, observed_frame_num - 1, :]
            x_t = np.expand_dims(x_t, axis=0)
            x_t = np.repeat(x_t, predicting_frame_num, axis=0)
            x_t = np.expand_dims(x_t, axis=0)
            preds = np.cumsum(preds, axis=1)
            preds = preds + x_t
            if p > best_p:
                best_pred = preds
        all_pred.append(best_pred)

    all_pred = np.reshape(all_pred, [x_val_.shape[0], 1, predicting_frame_num, 2])
    current_loss = calc_ade_meters(all_pred, x_val_)
    print(current_loss)
    if current_loss < best_val_loss:
        best_val_loss = current_loss
        cvae_mog.save_weights('CVAE_MoG_actev_best.h5')
        rounds_without_improvement = 0
    else:
        rounds_without_improvement += 1

    if rounds_without_improvement == patience:
        print("Early stopping...")
        break


#######################################Test prediction CVAE MoG################################3
cvae_mog.load_weights('CVAE_MoG_actev_best.h5')
all_pred = []
for data_point in tqdm(range(x_te.shape[0])):
    best_p = 0.0
    best_pred = []

    #for j in range(int(20 / n_centroid)):
    for i in range(n_centroid):
        u = u_p.eval()[:, i]
        l = lambda_p.eval()[:, i]
        z_sample = np.random.multivariate_normal(u, np.diag(l), (1,))
        p = get_posterior(z_sample, u_p.eval(), lambda_p.eval(), theta_p.eval())[i]
        obs_traj = np.expand_dims(y_te[data_point], axis=0)
        enc_condition = encoder.predict([obs_traj])
        preds = decoder.predict([z_sample, enc_condition])
        x_t = y_te_[data_point, observed_frame_num - 1, :]
        x_t = np.expand_dims(x_t, axis=0)
        x_t = np.repeat(x_t, predicting_frame_num, axis=0)
        x_t = np.expand_dims(x_t, axis=0)
        preds = np.cumsum(preds, axis=1)
        preds = preds + x_t
        if p > best_p:
            best_pred = preds
    all_pred.append(best_pred)


all_pred = np.reshape(all_pred, [x_te.shape[0], 1, predicting_frame_num, 2])
ade = calc_ade_meters(all_pred, x_te)
print("ADE: ", ade)
fde = calc_fde_meters(all_pred, x_te)
print("FDE: ", fde)

