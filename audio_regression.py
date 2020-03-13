import os
import math
import numpy as np
import scipy.linalg as la
import audio_features
import tensorflow as tf

from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_avec_pca_features(partition, pca, need_fit=False):
    features = StandardScaler().fit_transform(audio_features.get_features(partition))
    return pca.fit_transform(features) if need_fit else pca.transform(features)

def get_egemaps_pca_features(partition, pca, need_fit=False):
    features = StandardScaler().fit_transform(avec.get_features(partition, feature_type='egemaps'))
    return pca.fit_transform(features) if need_fit else pca.transform(features

pca = PCA(40)

f_train = get_avec_pca_features('training', pca, True)
# f_train = get_egemaps_pca_features('training', pca, True)
l_train = audio_features.get_labels("training")
f_dev = get_avec_pca_features('development', pca)                                                                    
# f_dev = get_egemaps_pca_features('development', pca)
l_dev = audio_features.get_labels("development")

# f_aug, l_aug = avec.augment_features(f_train, l_train, 1000)
# f_train = np.concatenate((f_train, f_aug), axis=0)
# l_train = np.concatenate((l_train, l_aug), axis=0)

pls = PLSRegression()
pls.fit(f_train, l_train)
pls_pred = pls.predict(f_dev).squeeze()

lr = LinearRegression()
lr.fit(f_train, l_train)
lr_pred = lr.predict(f_dev)

pred = np.round((pls_pred + lr_pred) / 2)
rmse = sqrt(mean_squared_error(l_dev, pred))
mae = mean_absolute_error(l_dev, pred)

print("RMSE = ", rmse)
print("MAE  = ", mae)

model = tf.keras.Sequential([
    layers.Dense(64, activation='sigmoid', input_shape=(40,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(16)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.MeanSquaredLogarithmicError(),
              metrics=['mae'])

model.fit(np.array(f_train), np.array(l_train), epochs=500)
model.evaluate(np.array(f_dev), np.array(l_dev))

pred_nn = np.array(tf.math.reduce_mean(model.predict(np.array(f_dev)).squeeze(), axis=1).numpy(), dtype=int)
pred_nn = np.array(np.round((pred_nn + pred) / 2))

rmse = sqrt(mean_squared_error(l_dev, pred_nn))
mae = mean_absolute_error(l_dev, pred_nn)

print ('RMSE NN: ', rmse)
print("MAE NN: ", mae)