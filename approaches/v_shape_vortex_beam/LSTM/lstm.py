from keras.layers import Dense, LSTM, Dropout, Masking
from keras.utils import to_categorical
from keras.initializers import Identity
from keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from utils import load, my_init, normalization
import math
from Const import Consts
from keras import optimizers

import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

from keras.callbacks import LearningRateScheduler
import keras.backend as K
def scheduler(epoch):
    if epoch % 5 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        if lr < 5e-8:
            return lr
        K.set_value(model.optimizer.lr, lr * 0.8)
        print("lr changed to {}".format(lr * 0.8))
    return K.get_value(model.optimizer.lr)

n_steps = Consts["length_mesh"]
n_inputs = Consts["width_mesh"] * Consts["rect_num"]
n_outputs = Consts["width_mesh"] * Consts["rect_num"]
n_hiddens = Consts["width_mesh"] * Consts["rect_num"]
n_layers = 2
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(n_steps, n_inputs)))
model.add(LSTM(units=n_outputs, kernel_initializer='orthogonal', recurrent_initializer='orthogonal'))

Optor = optimizers.Adam(lr=0.0001)

model.compile(loss='mse', optimizer= Optor)
print(1)


# data
print(2)
data_x, data_y = load()
data_y = normalization(data_y)

test_x = data_x[-Consts["test_amount"] * Consts["length_mesh"]:]
test_y = data_y[-Consts["test_amount"] * Consts["length_mesh"]:]
train_x = data_x[:-Consts["test_amount"] * Consts["length_mesh"]]
train_y = data_y[:-Consts["test_amount"] * Consts["length_mesh"]]
print(3)

model.summary()
reduce_lr = LearningRateScheduler(scheduler)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto', epsilon=0.0005)
history = model.fit(train_x, train_y, batch_size=128, epochs=200, verbose=2, validation_split=0.1, callbacks=[reduce_lr])

model.save("lstm_mode2500a.h5")

print(model.evaluate(test_x, test_y, batch_size=128, verbose=0))
