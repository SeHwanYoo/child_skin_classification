from glob import glob 
import numpy as np
import tensorflow as tf

files = []
for i in range(7, 10):
    # for key in train_dict.keys():
    img = glob(dataset + f'/H{str(i)}/*/*.jpg')
    files.extend(img) 
