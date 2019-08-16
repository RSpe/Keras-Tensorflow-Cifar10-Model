import tensorflow as tf
from tensorflow.keras import layers
import pickle
import tarfile
import numpy as np
import scipy as sc
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import matplotlib.pyplot as plt

def extract(targz):

    tar = tarfile.open("cifar-10-python.tar.gz")
    tar.extractall()
    tar.close

def unpickle(cifar):

    with open(cifar, "rb") as fo:
        data_batch = pickle.load(fo, encoding="bytes")
    return data_batch

def fix_input(data_batch):

    image_height = 32
    image_width = 32
    rgb_pixels = data_batch[b"data"].reshape(len(data_batch[b"labels"]), 3, image_width, image_height)
    labels = data_batch[b"labels"]

    return rgb_pixels, labels

def median_filter(pixels, window_size, rgb): #get rid of noise

    for i in range(len(pixels)):
        for j in range(rgb):
            final = sc.ndimage.filters.median_filter(pixels[i][j], size = (3, 3))
            pixels[i][j] = final

    return pixels

def histogram_eq(pixels, w, h, rgb): #adaptive, increase sharpness and decrease median filter blur

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    #print(pixels[0][1])
    for i in range(len(pixels)):
        for j in range(rgb):
            final = clahe.apply(pixels[i][j])
            pixels[i][j] = final

    #print(pixels[0][1])
    return pixels
    
def normalise(x_train, x_test):
    
    x_train = pixels.astype("float32")
    x_test = x_test.astype("float32")

    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train - mean)/(std + 1e-7)
    x_test = (x_test - mean)/(std + 1e-7)
    
    return x_train, x_test

def tf_reset(pixels, labels):
    
    tf.compat.v1.reset_default_graph()

    test_set = unpickle("cifar-10-batches-py/test_batch")
    test_pixels, test_labels = fix_input(test_set)

    x_train = pixels
    y_train = labels
    x_test = test_pixels
    y_test = test_labels

    x_train, x_test = normalise(x_train, x_test)

    return x_train, y_train, x_test, y_test

def tfk_model(x_train, y_train, x_test, y_test, num_classes):


    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
    
    model = tf.keras.models.Sequential()

    # Convolutional layer 1
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", input_shape = x_train.shape[1:]))
    model.add(tf.keras.layers.Activation("selu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    # Convolutional layer 2
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("selu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))

    # Convolutional layer 3
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("selu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())

    #Fully connected layer 1
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation("selu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    #Fully connected layer 2
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation("softmax"))

    model.summary()

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    datagen = ImageDataGenerator(rotation_range = 20, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)
    datagen.fit(x_train)

    batch_size = 32
    epochs = 100

    training = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), steps_per_epoch = 10000 / batch_size, epochs = epochs, validation_data=(x_test, y_test))
    
    final_score = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 1)

    print("Validation loss: ", final_score[0])
    print("Validation accuracy: ", final_score[1])

    return training

def plots(model):

    plt.plot(training.history["loss"])
    plt.plot(training.history["val_loss"])
    plt.title("Training loss and validation loss over time as the number of epochs increase")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.show()

    plt.plot(training.history["acc"])
    plt.plot(training.history["val_acc"])
    plt.title("Training accuracy and validation accuracy over time as the number of epochs increase")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training accuracy", "Validation accuracy"])
    plt.show()

if __name__ == "__main__":
    #extract("cifar-10-python.tar.gz")
    data = unpickle("cifar-10-batches-py/data_batch_1")
    pixels, labels = fix_input(data)
    #print(pixels[0][0])
    #median_filter(pixels, 3, 3)
    pixels = median_filter(pixels, 3, 3)
    pixels = histogram_eq(pixels, 32, 32, 3)
    x_train, y_train, x_test, y_test = tf_reset(pixels, labels)
    model = tfk_model(x_train, y_train, x_test, y_test, 10)
    plots(model)
    #print(pixels[0][0])