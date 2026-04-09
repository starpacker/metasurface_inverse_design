# -*- coding: utf-8 -*-

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 49
intermediate_dim2 = 49
epochs = 20
epsilon_std = 1.0
pixel_length = 28


# 首先需要改写 input layer 到 卷积层形状-------Encoder
inputs = Input(batch_shape = (batch_size, pixel_length,pixel_length,1))

x = Conv2D(32, (3, 3), activation='relu', padding='same', name="Conv_1")(inputs)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name="Conv_2")(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same', name="Conv_3")(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name="Conv_4")(x)

x = Conv2D(1, (3, 3), activation='relu', padding='same', name="Conv_5")(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same', name="Conv_6")(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Flatten()(x)
h = Dense(intermediate_dim, activation='relu', name="Dense_1")(x)
z_mean = Dense(latent_dim, name="Dense_2a")(h)
z_log_var = Dense(latent_dim, name="Dense_2b")(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
# 然后改写 Decoder 为 CNN 的模式
#decoder_h = Dense(intermediate_dim, activation='relu')
#decoder_mean = Dense(original_dim, activation='sigmoid')
#h_decoded = decoder_h(z)
#x_decoded_mean = decoder_mean(h_decoded)

h_decoded = Dense(intermediate_dim2, activation='relu', name="Dense_3")(z)

print(h_decoded)
h = Reshape((7, 7))(h_decoded)
myexpanddim= Lambda(lambda x: K.expand_dims(x, axis=3))
# x=K.tf.reshape(h_decoded,(-1,7,7,1))
x=myexpanddim(h)
print(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same',name="Conv_7")(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same',name="Conv_8")(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same',name="Conv_9")(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same',name="Conv_10")(x)

x = Conv2D(1, (3, 3), activation='relu', padding='same',name="Conv_11")(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same',name="Conv_12")(x)
x = UpSampling2D((2, 2))(x)

x_decoded_mean = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name="Conv_13")(x)


# instantiate VAE model
vae = Model(inputs, x_decoded_mean)
encoder = Model(inputs, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,)) # 输入

h_decoded = Dense(intermediate_dim2, activation='relu', name="Dense_4")(x)

x = Reshape((7, 7))(h_decoded)
myexpanddim= Lambda(lambda x: K.expand_dims(x, axis=3))
# x=K.tf.reshape(h_decoded,(-1,7,7,1))
x=myexpanddim(h)
x = Conv2D(32, (3, 3), activation='relu', padding='same',name="Conv_14")(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same',name="Conv_15")(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(1, (3, 3), activation='relu', padding='same',name="Conv_18")(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same',name="Conv_19")(x)
x = UpSampling2D((2, 2))(x)

_x_decoded_mean = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name="Conv_20")(x)
generator = Model(decoder_input, _x_decoded_mean)

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
