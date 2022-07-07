import cv2 
import tensorflow as tf
from glob import glob
import numpy as np
import main

def label_2_index(lbl, label_dict):
    return label_dict[lbl]

def index_2_label(idx, label_dict):
    key = [keys for keys in label_dict if (label_dict[keys] == idx)]
    return key

def train_generator(images, labels, aug=False):
    
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        # print(img)
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (main.n_res, main.n_res))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # idx = f.split('\\')[1].split('/')[2]
        # lbl = tf.keras.utils.to_categorical(lbl, len(train_dict))

        yield (img, lbl)    
        
        # if lower than base num, should apply data augmentation
        # if base_num <= int(train_dict[idx]):
        if aug:

            # Btight 
            random_bright_tensor = tf.image.random_brightness(img, max_delta=128)
            random_bright_tensor = tf.clip_by_value(random_bright_tensor, 0, 255)
            random_bright_image = tf.keras.preprocessing.image.array_to_img(random_bright_tensor)
            yield (random_bright_tensor, lbl) 
    
            # rotation
            rotated_img = tf.image.rot90(img)        
            yield (rotated_img, lbl) 
            
            # # curmix 
            # cutmixed_img, cutmixed_lbl = cutmix(img, lbl)
            # yield (cutmixed_img, cutmixed_lbl)
            
                
def test_generator(images, labels):
    
    for img, lbl in zip(images, labels):
        
        img = img[0].decode('utf-8')
        
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (main.n_res, main.n_res))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        yield (img, lbl)    
            
def create_dataset(images, labels, d_type='train', aug=False):
    
    if d_type == 'test':
        return tf.data.Dataset.from_generator(test_generator, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([main.n_res, main.n_res, 3]), tf.TensorShape([1])),
                                              args=[images, labels])
        
    else:
        return tf.data.Dataset.from_generator(train_generator, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([main.n_res, main.n_res, 3]), tf.TensorShape([1])),
                                              args=[images, labels, aug])
        
        
def create_train_list(dataset):
    images = []
    for i in range(6):
        img = glob(dataset + f'/H{str(i)}/*/*.jpg')
        images.extend(img) 

    img = glob(dataset + '/H9/*/*.jpg')
    images.extend(img) 
    
    label_dict = dict()

    lbls = [] 
    for img in images:
        lbl = img.split('/')[-1].split('\\')[0]
        lbls.append(lbl)
        
    lbls = np.unique(lbls)

    label_dict = {lbl : i for i, lbl in enumerate(lbls)}
        
    labels = [] 
    for img in images:
        lbl = img.split('/')[-1].split('\\')[0]
        lbl = label_2_index(lbl, label_dict)
        labels.append(lbl)
        
    labels = np.reshape(labels, [-1, 1])
    
    
    return images, labels 
    
    