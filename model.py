import tensorflow
import pickle
import tarfile
import numpy as np
import scipy as sc
import cv2

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

def tf(w, h, rgb):
    
    tf.reset_default_grapgh

if __name__ == "__main__":
    extract("cifar-10-python.tar.gz")
    data = unpickle("cifar-10-batches-py/data_batch_1")
    pixels, label, names = fix_input(data)
    #print(pixels[0][0])
    median_filter(pixels, 3, 3)
    histogram_eq(pixels, 32, 32, 3)
    normalise(pixels, 3)
    tf(32, 32, 3)
    #print(pixels[0][0])