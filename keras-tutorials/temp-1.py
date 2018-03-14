import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import Adam,SGD

np.random.seed(35)
# X_train, y_train), (X_test, y_test) = mnist.load_data()

x_data = pd.read_csv('x_data.dat',header=None)
y_data = pd.read_csv('x_test.dat',header=None)
x_data=x_data.T
y_data=y_data.T
num=111
x = np.linspace(0, 1, 150)
y = x_data[num]
z = y_data[num]
plt.plot(x,y)  
plt.plot(x,z)

x_data=x_data.T
y_data=y_data.T
# defining autoencoder and it's layer
input_signal = Input(shape=(150,))
encoded = Dense(150, activation='relu')(input_signal)
#encoded1 = Dense(150, activation='linear')(encoded)
#decoded = Dense(150, activation='relu')(encoded1)
decoded1 = Dense(150, activation='softmax')(encoded)
autoencoder = Model(input_signal, decoded1)
adam = Adam(lr=0.9)
autoencoder.compile(optimizer=adam, loss='categorical_crossentropy')

print (autoencoder.summary())

# traning auto encoder
autoencoder.fit(x_data, y_data,
                epochs=10,
                batch_size=132)

x_data=x_data.T
h=x_data[1]
h=h.as_matrix()
h.shape=(150,1)
h=h.T
mm=autoencoder.predict(h[0])
