import numpy as np
import random
from tensorflow import keras
from tqdm import tqdm
import mathematics.Algebra as al

data = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
with open("answer.txt",mode="w") as file:
    for x in range(train_images.shape[0]):
        file.write(str(train_labels[x]))
        file.write("\n")
        for n in range(28):
            for p in range(28):
                #file.write(str(train_images[x][n][p]))
                #file.write("\n")
                pass
        if x %1000==0:
            print(x)