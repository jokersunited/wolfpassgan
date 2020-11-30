import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
import numpy as np

charmap = 55
pass_len = 10
pass_shape = (charmap, pass_len)

noise_shape = (100,)

model = Sequential()
model.add(Dense(256, input_shape=noise_shape))
model.add(Dense(np.prod(pass_shape), activation='tanh'))
model.add(Reshape(pass_shape))
model.summary()

noise = Input(shape=noise_shape)
passwords = model(noise)

output = Model(noise, passwords)

#Generation
noise = np.random.normal(0, 1, (1, 100))
gen_stuff = output.predict(noise)

def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

print(gen_stuff)

sample = np.argmax(gen_stuff, axis=1)
sample2 = np.argmax(softmax(gen_stuff, 55), axis=1)

print(np.sum(gen_stuff, axis=1))
print(np.sum(softmax(gen_stuff, 55), axis=1))

password = ""
password2 = ""
for x in sample[0]:
  password += chr(x+65)
print(password)
for x in sample2[0]:
  password2 += chr(x+65)
print(password2)