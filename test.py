import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv1D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.models import Sequential, Model
import numpy as np

charmap = 55
pass_len = 10
pass_shape = (charmap, pass_len)

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
    model.add(Conv1D(filters=100, kernel_size=3, activation='relu', input_shape=(10, 1)))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    intput_pass = Input(shape=(10, 1))
    validity = model(intput_pass)

    return Model(intput_pass, validity)

#Testing
generator = Generator()
discriminator = Discriminator()

noise = np.random.normal(0, 1, (1, 100))
gen_stuff = generator.predict(noise)

def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

print(gen_stuff)

sample = np.argmax(gen_stuff, axis=1)
# sample2 = np.argmax(softmax(gen_stuff, 55), axis=1)

print(sample)

discriminate = discriminator.predict(sample)
print("output: " + str(discriminate))

# print(np.sum(gen_stuff, axis=1))
# print(np.sum(softmax(gen_stuff, 55), axis=1))

noise2 = np.random.normal(0, 1, (1000, 100))

# Generate a half batch of fake images
gen_imgs = generator.predict(noise2)
print(gen_imgs)

for gen in gen_imgs:
  sample = np.argmax(gen, axis=0)
  password = ""
  for x in sample:
      password += chr(x+65)
  print(password)



# password = ""
# password2 = ""
# for x in sample[0]:
#   password += chr(x+65)
# print(password)
# for x in sample2[0]:
#   password2 += chr(x+65)
# print(password2)