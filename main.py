import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv1D, Multiply, Permute
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.backend import argmax
import numpy as np
from utils import *

layer_dim = 128
pass_len = 10
good_list, charmap, inv_charmap = load_dataset("10-million-password-list-top-1000000.txt", pass_len)
pass_shape = (len(inv_charmap), pass_len)

noise_len = 256
noise_shape = (noise_len,)

def Conversion(one_dim):
  two_dim = np.zeros((len(inv_charmap),10))
  for x in range (len(one_dim)):
    two_dim[one_dim[x]][x] = 1
  return two_dim


class ResBlock(tf.keras.Model):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = tf.keras.Sequential([
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 5, padding='same'),
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 5, padding='same'),
        ])

    def call(self, input, **kwargs):
        output = self.res_block(input)
        return input + (0.3 * output)

def Generator(dim):
    model = Sequential()
    model.add(Dense(256, activation='linear', input_shape=noise_shape))
    model.add(Reshape((-1, 2, 128)))
    model.add(ResBlock(dim))
    model.add(ResBlock(dim))
    model.add(ResBlock(dim))
    model.add(ResBlock(dim))
    model.add(ResBlock(dim))
    model.add(Reshape((1, 32, 8)))
    model.add(Conv1D(64, 32, 1, padding='valid'))
    model.add(Flatten())
    model.add(Dense(np.product(pass_shape), activation='linear'))
    model.add(Softmax(axis=1))
    model.add(Reshape(pass_shape))

    noise = Input(shape=noise_shape)
    password = model(noise)

    model.summary()
    return Model(noise, password)


def Discriminator(dim):
    model = Sequential()
    model.add(Dense(128, input_shape=pass_shape))
    model.add(Flatten())
    model.add(Reshape((1, 32, int(len(inv_charmap) * 128 / 32))))
    model.add(Conv1D(dim, 32, 1, padding='valid'))
    model.summary()
    model.add(ResBlock(dim))
    model.add(ResBlock(dim))
    model.add(ResBlock(dim))
    model.add(ResBlock(dim))
    model.add(ResBlock(dim))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))

    passw = Input(shape=pass_shape)
    valid = model(passw)

    model.summary()

    return Model(passw, valid)

#Saving samples from generator
def save(epochs):
  noise = np.random.normal(0, 1, (noise_len, noise_len))
  output = generator.predict(noise)
  output = np.argmax(output, axis=1)
  with open("progress/" + str(epochs) + "_sample_100.txt", "w") as f:
    for out in output:
      out_pass = ""
      for x in out:
        out_pass += inv_charmap[x]
      f.write(out_pass+"\n")

#The whole training process
def training(epochs, batch_size, sample_interval):
  for epoch in range (0, epochs):
    # Generate a half batch of fake images
    noise = np.random.normal(0, 1, (batch_size, noise_len))
    gen_pass = generator.predict(noise)

    # #format the output of the generator to train the discriminator
    # bad_list = []
    # for gen in gen_imgs:
    #   sample = np.argmax(gen, axis=0)
    #   bad_list.append(sample)

    # bad_list = np.array(bad_list)

    #Take a random sample of the good batch to train
    idx = np.random.randint(0, len(good_list), batch_size)
    good_list_t = np.array(good_list)[idx]
    good_list_t = np.array([Conversion(x) for x in good_list_t])

    #Train discriminator
    d_loss_real = discriminator.train_on_batch(good_list_t, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_pass, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    #Train generator
    noise = np.random.normal(0, 1, (batch_size, noise_len))
    g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

    print("\n%d Iterations [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    if epoch % sample_interval == 0:
      save(epoch)

#Things for ipynb
optimizer = Adam(0.0002, 0.5)

generator = Generator(layer_dim)
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

discriminator = Discriminator(layer_dim)
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

#Defining input for combined model
z = Input(shape=noise_shape)
gen_pass = generator(z)

discriminator.trainable = False

#Pass output of generator to discriminator in the combined model
validity = discriminator(gen_pass)

#Finalize the combined model
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

training(1000, 64, 50)