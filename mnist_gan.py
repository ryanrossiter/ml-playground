import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_train = x_train.reshape(60000, 784)

    return (x_train, y_train)

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def create_generator():
    gen = Sequential()
    gen.add(Dense(units=256, input_dim=100))
    gen.add(LeakyReLU(0.2))

    gen.add(Dense(units=512))
    gen.add(LeakyReLU(0.2))

    gen.add(Dense(units=1024))
    gen.add(LeakyReLU(0.2))

    gen.add(Dense(units=784, activation='tanh'))

    gen.compile(loss='binary_crossentropy', optimizer=adam_optimizer())

    return gen

def create_discriminator():
    dis = Sequential()
    dis.add(Dense(units=1024, input_dim=784))
    dis.add(LeakyReLU(0.2))
    dis.add(Dropout(0.3))

    dis.add(Dense(units=512))
    dis.add(LeakyReLU(0.2))
    dis.add(Dropout(0.3))

    dis.add(Dense(units=256))
    dis.add(LeakyReLU(0.2))

    dis.add(Dense(units=1, activation='sigmoid'))

    dis.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return dis

def create_gan(gen, dis):
    dis.trainable = False
    gan_input = Input(shape=(100,))
    x = gen(gan_input)
    gan_output = dis(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def plot_images(epoch, gen, examples=100, dim=(10,10), figsize=(10,10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = gen.predict(noise).reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/gan_gen_%d.png' % epoch)

def train(epochs=1, batch_size=128):
    (x_train, y_train) = load_data()
    batch_count = x_train.shape[0] // batch_size + 1

    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(generator, discriminator)

    for e in range(1, epochs+1):
        print("Epoch %d" % e)

        for _ in tqdm(range(batch_count)):
            noise = np.random.normal(0, 1, [batch_size, 100])

            generated_images = generator.predict(noise)

            image_batch = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]

            x = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(x, y_dis)

            discriminator.trainable = False
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gan = np.ones(batch_size)

            gan.train_on_batch(noise, y_gan)

        if e == 1 or e % 20 == 0:
            plot_images(e, generator)

train(400)

