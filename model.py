import tensorflow as tf
from tensorflow.keras import layers
import pickle
import tarfile
import numpy as np
import scipy as sc
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

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
    names = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
    
    return rgb_pixels, labels, names

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
    
def normalise(pixels, rgb):
    
    minv = np.min(pixels)
    maxv = np.max(pixels)
    pixels = (pixels - minv) / (maxv - minv)
    
    return pixels

def tf_reset(pixels, labels):
    
    tf.compat.v1.reset_default_graph()

    test_set = unpickle("cifar-10-batches-py/test_batch")
    test_pixels, test_labels, test_names = fix_input(test_set)

    x_train = pixels
    y_train = labels
    x_test = test_pixels
    y_test = test_labels

    return x_train, y_train, x_test, y_test

def tfk_model(x_train, y_train, x_test, y_test, num_classes):


    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
    
    model = tf.keras.models.Sequential()

    # Convolutional layer 1
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", input_shape = x_train.shape[1:]))
    model.add(tf.keras.layers.Activation("elu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Convolutional layer 2
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", input_shape = x_train.shape[1:]))
    model.add(tf.keras.layers.Activation("elu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    # Convolutional layer 3
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", input_shape = x_train.shape[1:]))
    model.add(tf.keras.layers.Activation("elu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (3, 3)))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())

    #Fully connected layer 1
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation("elu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    #Fully connected layer 2
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation("softmax"))

    model.summary()

    model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"])

    model.fit(x_train, y_train, epochs = 10, batch_size = 128)
    score = model.evaluate(x_test, y_test, batch_size = 128)

    print("Tests loss: ", score[0])
    print("Tests accuracy: ", score[1])

if __name__ == "__main__":
    #extract("cifar-10-python.tar.gz")
    data = unpickle("cifar-10-batches-py/data_batch_1")
    pixels, labels, names = fix_input(data)
    #print(pixels[0][0])
    #median_filter(pixels, 3, 3)
    #histogram_eq(pixels, 32, 32, 3)
    pixels = normalise(pixels, 3)
    x_train, y_train, x_test, y_test = tf_reset(pixels, labels)
    tfk_model(x_train, y_train, x_test, y_test, len(names))
    #print(pixels[0][0])