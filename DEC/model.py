import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utils import data_loader
from sklearn.cluster import KMeans
from utils import save_images, make_dirs, acc, nmi, ari


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(500, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(500, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(2000, activation=tf.nn.relu)
        self.bottleneck = tf.keras.layers.Dense(10)

    def call(self, inp):
        x_reshaped = self.flatten_layer(inp)
        x = self.dense1(x_reshaped)
        x = self.dense2(x)
        x = self.dense3(x)
        latent = self.bottleneck(x)
        return latent
    
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense4 = tf.keras.layers.Dense(2000, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(500, activation=tf.nn.relu)
        self.dense6 = tf.keras.layers.Dense(500, activation=tf.nn.relu)
        self.dense_final = tf.keras.layers.Dense(784)
    
    def call(self, inp):
        x = self.dense4(inp)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense_final(x)
        return x
    
class DEC(object):
    def __init__(self, config):
        self.config = config
        self.enc = Encoder()
        self.dec = Decoder()
        self.alpha = 1.0
        self.latent_dim = 10
        #self.optim = tf.keras.optimizers.Adam(self.config.learning_rate)
        self.pre_optim = tf.keras.optimizers.SGD(lr=1, momentum=0.9)
        self.fin_optim = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        self.global_step = tf.Variable(0, trainable=False)
        self.global_epoch = tf.Variable(0, trainable=False)
        self.cluster_centers = tf.Variable(tf.zeros([self.config.n_clusters, self.config.latent_dim]), trainable=True)
        self.x, self.y, self.trainloader = data_loader(config)

    def pretrain(self):
        print('Pretraining start!')

        for epoch in tqdm(range(200)):
            epoch_loss = []
            for x_batch, _ in self.trainloader:
                with tf.GradientTape() as tape:
                    z = self.enc(x_batch)
                    x_rec = self.dec(z)
                    batch_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.keras.layers.Flatten()(x_batch), x_rec))
                             
                t_vars = self.enc.trainable_variables + self.dec.trainable_variables
                enc_grads = tape.gradient(batch_loss, t_vars)
                self.pre_optim.apply_gradients(zip(enc_grads, t_vars))
                epoch_loss.append(batch_loss) 
            print('epoch_loss:{:.4f}'.format(tf.reduce_mean(epoch_loss).numpy()))
            
        print('Pretraining finish!')

    def initialize(self):
        z = np.array([]).astype('float32').reshape(0, self.latent_dim)
        true = np.array([])
        for x_batch, labels in self.trainloader:
            latent = self.enc(x_batch)
            z = np.vstack((z, latent))
            true = np.append(true, labels)
        kmeans = KMeans(n_clusters=10, n_init=20).fit(z)
        self.cluster_centers.assign(kmeans.cluster_centers_)
        
        pred = kmeans.predict(z)        
        acc_ = acc(true, pred)
        nmi_ = nmi(true, pred)
        ari_ = ari(true, pred)
        print('acc:{}, nmi:{}, ari:{}'.format(acc_, nmi_, ari_))
        
    def cluster_assign(self, z):
        z = tf.expand_dims(z, axis=1)
        q = 1.0 + (tf.reduce_sum(tf.math.square(z - self.cluster_centers), axis=2) / 1.)
        q = q ** (- (1. + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q
    
    def target_distribution(self, q):
        weight = q ** 2 / tf.reduce_sum(q, axis=0)
        p = weight / tf.reduce_sum(weight, axis=1, keepdims=True)
        return p

    def finetune(self):
        for epoch in tqdm(range(20)):
            z = self.enc(self.x)
            q = self.cluster_assign(z)
            p = self.target_distribution(q)
            
            epoch_loss = []
            pred = np.array([])
            true = np.array([])
            for index, (x_batch, labels) in enumerate(self.trainloader):
                with tf.GradientTape() as tape:
                    latent = self.enc(x_batch)
                    q_ = self.cluster_assign(latent)
                    kl = tf.keras.losses.KLDivergence()
                    batch_loss = kl(p[index * self.config.batch_size: min((index+1) * self.config.batch_size, self.x.shape[0])], q_)    
                t_vars = self.enc.trainable_variables + [self.cluster_centers]
                t_grads = tape.gradient(batch_loss, t_vars)
                self.fin_optim.apply_gradients(zip(t_grads, t_vars))
                epoch_loss.append(batch_loss)

                y_pred = np.argmax(q_, axis=1)
                pred = np.append(pred, y_pred)
                true = np.append(true, labels)

            acc_ = acc(true, pred)
            nmi_ = nmi(true, pred)
            ari_ = ari(true, pred)
            print('epoch:{}, epoch_loss:{:.4f}, acc:{:.4f}, nmi:{:.4f}, ari:{:.4f}'.format(
                   epoch, tf.reduce_mean(epoch_loss).numpy(), acc_, nmi_, ari_))