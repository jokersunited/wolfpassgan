import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv1D, multiply, add
from keras.layers import BatchNormalization
from keras.activations import linear, relu
from keras.layers.advanced_activations import LeakyReLU, ReLU, Softmax
from keras.models import Sequential, Model
import numpy as np

charmap = 55
pass_len = 10
pass_shape = (charmap, pass_len)

noise_shape = (100,)

layer_dim = 1
kernel_size = 55

def ResBlock(model):
    block = model
    block.add(ReLU())
    # block = relu(block)
    block.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu', input_shape=(10, layer_dim), padding='same'))
    block.add(ReLU())
    # block = relu(block)
    block.add(Conv1D(filters=1, kernel_size=kernel_size, activation='relu', input_shape=(10, layer_dim), padding='same'))
    # return multiply([model, block])
    return block

#Generator Model

def Generator():
    model = Sequential()
    model.add(Flatten(input_shape=noise_shape))
    # model.add(Dense(10, input_shape=noise_shape))
    # model.add(Dense(256, input_shape=noise_shape))
    # model.add(Dense(np.prod(pass_shape), activation='sigmoid'))
    # model.add(Conv1D(filters=1, kernel_size=10, activation='relu', input_shape=(10, 1), padding='same'))
    
    model = linear(model)

    model.add(Reshape((100, layer_dim)))
    model = ResBlock(model)
    model = ResBlock(model)
    model = ResBlock(model)
    model = ResBlock(model)
    model = ResBlock(model)

    model.add(Conv1D(filters=10, kernel_size=kernel_size, activation='relu', input_shape=(10, layer_dim), padding='same'))
    model.add(Softmax(axis=1))
    model.summary()

    noise = Input(shape=noise_shape)
    passwords = model(noise)

    return Model(noise, passwords)

#Discriminator Model

def Discriminator():
    model = Sequential()
    
    # model.add(Flatten(input_shape=pass_shape))
    model.add(Conv1D(filters=1, kernel_size=10, activation='relu', input_shape=(10, layer_dim), padding='same'))
    model = ResBlock(model)
    model = ResBlock(model)
    model = ResBlock(model)
    model = ResBlock(model)
    model = ResBlock(model)
    model.add(Reshape((10, layer_dim)))
    model = linear(model)
    
    # model.add(Dense(256))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Flatten())
    # model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=(10, 1))
    validity = model(img)

    return Model(img, validity)

#Testing
generator = Generator()
discriminator = Discriminator()

noise = np.random.normal(0, 1, (1, 100))
gen_stuff = generator.predict(noise)

sample = np.argmax(gen_stuff, axis=1)

discriminate = discriminator.predict(sample)
print(discriminate)

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

print(np.sum(gen_stuff, axis=1))
# print(np.sum(softmax(gen_stuff, 55), axis=1))

password = ""
password2 = ""
for x in sample[0]:
  password += chr(x+65)
print(password)
# for x in sample2[0]:
#   password2 += chr(x+65)
# print(password2)