import tensorflow
import pickle
import tarfile
import numpy as np

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
    rgb_pixels = data_batch[b"data"].reshape(len(data_batch[b"labels"]), 3, image_width, image_height).transpose(0, 2, 3, 1)
    labels = data_batch[b"labels"]
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    return rgb_pixels, labels, names

#def preprocess(pixels):



if __name__ == "__main__":
    extract("cifar-10-python.tar.gz")
    data = unpickle("cifar-10-batches-py/data_batch_1")
    #print(output)
    pixels, lablel, names = fix_input(data)
    preprocess(pixels)