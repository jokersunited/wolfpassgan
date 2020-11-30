import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv1D, Multiply
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from utils import *


pass_len = 10
good_list, charmap, inv_charmap = load_dataset("../10-million-password-list-top-1000000.txt", pass_len)
pass_shape = (len(inv_charmap[1:]), pass_len)

noise_shape = (100,)

#Generator Model
def Generator():
  model = Sequential()
  model.add(Flatten(input_shape=noise_shape))
  model.add(Dense(256))
  model.add(Dense(np.prod(pass_shape), activation='tanh'))
  model.add(Reshape(pass_shape))
  model.add(Softmax(axis=1))
  model.summary()

  noise = Input(shape=noise_shape)
  passwords = model(noise)
  return Model(noise, passwords)

#Discriminator Model
def Discriminator():
    model = Sequential()
    model.add(Conv1D(filters=5*pass_len, kernel_size=5, activation='relu', input_shape=(pass_len, 1)))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    intput_pass = Input(shape=(pass_len, 1))
    validity = model(intput_pass)

    return Model(intput_pass, validity)

#Testing
generator = Generator()

optimizer = Adam(0.0002, 0.5)
discriminator = Discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

for epoch in range (0, 100):
  noise = np.random.normal(0, 1, (1, 100))
  gen_stuff = generator.predict(noise)

  # Generate a half batch of fake images
  noise2 = np.random.normal(0, 1, (10000, 100))
  gen_imgs = generator.predict(noise2)

  bad_list = []
  for gen in gen_imgs:
    sample = np.argmax(gen, axis=0)
    bad_list.append(sample)

  bad_list = np.array(bad_list)

  idx = np.random.randint(0, len(good_list), 10000)
  good_list_2 = np.array(good_list)[idx]

  # print(bad_list)
  # print(good_list)
  # print(good_list_2)

  d_loss_real = discriminator.train_on_batch(good_list_2, np.ones((10000, 1)))
  d_loss_fake = discriminator.train_on_batch(bad_list, np.zeros((10000, 1)))

  d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

  # print(np.array([good_list[6347]]))
  # print(gen_stuff2)
  # exit()

  print(d_loss) 

  def testing():
    print("\n=== Test Case ===")
    number = np.random.randint(0,99999)
    print("good prediction: ")
    print(discriminator.predict(np.array([good_list[number]]))[0][0])
    good_pass = ""
    for x in good_list[number]:
      good_pass += inv_charmap[x]
    print(good_pass)

    noise = np.random.normal(0, 1, (1, 100))
    gen_stuff2 = generator.predict(noise)
    gen_stuff2 = np.argmax(gen_stuff2, axis=1)

    print("bad prediction: ")
    print(discriminator.predict(gen_stuff2)[0][0])
    bad_pass = ""
    for x in gen_stuff2[0]:
      bad_pass += inv_charmap[x]
    print(bad_pass)

  testing()
  print ("\n%d [D loss: %f, acc.: %.2f%%] [G loss: 0]" % (epoch, d_loss[0], 100*d_loss[1]))
# password = ""
# password2 = ""
# for x in sample[0]:
#   password += chr(x+65)
# print(password)
# for x in sample2[0]:
#   password2 += chr(x+65)
# print(password2)