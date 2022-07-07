import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten

import tensorflow_addons as tfa
import cv2
import os 
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
import random 
import math
import time
import pandas as pd

# user defined libs
# import models
# import dataset_generator
# from datasets import create_train_list
N_RES = 256 
N_BATCH = 16 
# PATH = 'C:/Users/user/Desktop/datasets/Child Skin Disease'
PATH = '../../datasets/Child Skin Disease'
dataset_path = os.path.join(PATH, 'Total_Dataset')

# Train & test set
min_num = 100
max_num = 3000 
base_num = 1000 

name_dict = {
    'Depressed scar' : 'Acne scar', 
    'Acquired tufted hemangioma' : 'Acquired tufted angioma', 
    'Cyst' : 'Epidermal cyst', 
    'Infantile hemangioma' : 'Hemangioma',
    'ILVEN': 'Inflammatory linear verrucous epidermal nevus'
}


AUTOTUNE = tf.data.AUTOTUNE

def label_2_index(lbl, label_dict):
    return label_dict[lbl]

def index_2_label(idx, label_dict):
    key = [keys for keys in label_dict if (label_dict[keys] == idx)]
    return key

def train_generator(images, labels, aug=False):
    
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_RES, N_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

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
        img = cv2.resize(img, (N_RES, N_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        yield (img, lbl)    
            
def create_dataset(images, labels, d_type='train', aug=False):
    
    if d_type == 'test':
        return tf.data.Dataset.from_generator(test_generator, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([N_RES, N_RES, 3]), tf.TensorShape([1])),
                                              args=[images, labels])
        
    else:
        return tf.data.Dataset.from_generator(train_generator, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([N_RES, N_RES, 3]), tf.TensorShape([1])),
                                              args=[images, labels, aug])
        
def create_all_dict(dataset, min_num, max_num):
    all_dict = dict() 
    
    for i in range(10):
        files = os.listdir(os.path.join(dataset, f'H{i}'))
        
        for f in files:
            # imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
            imgs = glob(f'{dataset}/H{i}/{f}/*.jpg')

            # class 통합 관련 내용 변경
            if f in name_dict: 
                f = name_dict[f]
            
            if f not in all_dict:
                all_dict[f] = len(imgs) 
            else:
                all_dict[f] += len(imgs)

    new_all_dict = all_dict.copy()
    # 데이터 정제
    for key, val in all_dict.items():
        if val < min_num:
            del new_all_dict[key]

        if val > max_num:
            new_all_dict[key] = max_num
                
    return new_all_dict

        
def create_train_list(dataset, all_dict):
    images = []
    for i in range(6):

        for key, val in all_dict.items():
            img = glob(dataset + f'/H{str(i)}/{key}/*.jpg')
            images.extend(img)

        for key, val in name_dict.items():
            img = glob(dataset + f'/H{str(i)}/{key}/*.jpg')
            images.extend(img)

        
    # 전남대 추가
    for key, val in all_dict.items(): 
        img = glob(dataset + '/H9/{key}/*.jpg')
        images.extend(img) 

    for key, val in name_dict.items():
        img = glob(dataset + f'/H9/{key}/*.jpg')
        images.extend(img)

    # 고른 데이터 분배를 위한 random shuffle
    random.shuffle(images)

    # max 데이터 처리
    # count 를 돌면서 count
    count_all_dict = all_dict.copy() 

    train_images = []
    for idx_imgs, val_imgs in enumerate(images):

        # class 통합 관련 내용 변경
        classes = val_imgs.split('/')[-2]
        if classes in name_dict:
            if count_all_dict[name_dict[classes]] > 0:
                count_all_dict[name_dict[classes]] -= 1
                # print(images[idx_imgs])
                # replace_images = val_imgs.replace(classes, name_dict[classes])
                # train_images.append(replace_images)
                train_images.append(val_imgs)

            else:
                continue

        else:
            if count_all_dict[classes] > 0:
                count_all_dict[classes] -= 1
                train_images.append(val_imgs)
            else:
                continue


    train_labels = [] 
    for img in train_images:
        # lbl = img.split('/')[-1].split('\\')[0]
        lbl = img.split('/')[-2]

        # 변경/통합 버전으로 label 처리
        if lbl in name_dict:
            lbl = name_dict[lbl]

        lbl = label_2_index(lbl, all_dict)
        train_labels.append(lbl)
        
    train_images = np.reshape(train_images, [-1, 1])
    train_labels = np.reshape(train_labels, [-1, 1])
    
    
    return train_images, train_labels
    

def get_dropout(input_tensor, p=0.3, mc=False):
    if mc: 
        layer = Dropout(p, name='top_dropout')
        return layer(input_tensor, training=True)
    else:
        return Dropout(p, name='top_dropout')(input_tensor, training=False)

def run_expriment(model_name, train_dataset, val_dataset, kfold=0, res=256, classes=10, batch_size=32, mc=False, epochs=100): 
    
    if model_name == 'efficient':
        # base_model = keras.applications.EfficientNetB0(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model = keras.applications.EfficientNetB7(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = True
        
        inputs = keras.Input(shape=(res, res, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        x = get_dropout(x, mc)
        x = keras.layers.Dense(classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    # VGG16 
    else:
        base_model = keras.applications.VGG16(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = True
        
        inputs = keras.Input(shape=(res, res, 3))
        x = base_model(inputs)
        x = keras.layers.Flatten(name = "avg_pool")(x) 
        x = keras.layers.Dense(512, activation='relu')(x)
        x = get_dropout(x, mc)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        

    sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'models/{model_name}_mc-{str(mc)}_bs-{batch_size}_{time.strftime("%Y%m%d-%H%M%S")}.h5'), 
                                             monitor='val_accuracy', 
                                             verbose=0, 
                                             save_best_only=True,
                                             save_weights_only=True, 
                                             mode='max', 
                                             save_freq='epoch'), 
          tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                           patience = 4, 
                                           min_delta = 0.01)
          ]

    
    LR = 0.0001
    # steps_per_epoch = len(train_images) // batch_size
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LR, steps_per_epoch*30, 0.1, True)
    
    # sgd = tf.keras.optimizers.SGD(0.01)
    # moving_avg_sgd = tfa.optimizers.MovingAverage(sgd)
    
    
    model.compile(loss='sparse_categorical_crossentropy', 
                #   optimizer = moving_avg_sgd, 
                  optimizer = tf.keras.optimizers.Adam(LR), 
                #   optimizer = tf.keras.optimizers.Adam(lr_schedule), 
                  metrics=['accuracy'])
    
    hist = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    # class_weight=class_weights, 
                    verbose=1,
                    shuffle=True,
                    callbacks=[sv])
    
    # histories.append(hist)
    
    return model, hist



if __name__ == '__main__':
    
    all_dict = create_all_dict(dataset_path, min_num, max_num)
    N_CLASSES = len(all_dict)
    # print(all_dict)

    train_images, train_labels = create_train_list(dataset_path, all_dict)
    
    
    # print(f'train_images : {train_images}')
    # print(f'train_labels : {train_labels}')

    kfold = 0 
    for skf_num in range(5, 11):
        skf = StratifiedKFold(n_splits=skf_num)
        
        for train_idx, valid_idx in skf.split(train_images, train_labels):
            
            # print(f'train_images : {train_images[train_idx]}')
            # print(f'train_labels : {train_labels[train_idx]}')
            
            train_dataset = create_dataset(train_images[train_idx], train_labels[train_idx], aug=True) 
            valid_dataset = create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
        
            train_dataset = train_dataset.batch(N_BATCH).shuffle(1000)
            valid_dataset = valid_dataset.batch(N_BATCH).shuffle(1000)
        
            model, hist = run_expriment('efficient', train_dataset, valid_dataset, res=N_RES, classes=N_CLASSES, batch_size=N_BATCH, mc=False, epochs=100)

            # evaluation
            # PATH = '../../datasets/Child Skin Disease'
            # model.save(f'/home/ubuntu/models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{kfold}.h5')

            # # import pandas as pd
            # hist_df = pd.DataFrame(hist.history)
            # with open(f'/home/ubuntu/models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{kfold}.csv', mode='w') as f:
            #     hist_df.to_csv(f)

            model.save(f'../../models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{kfold}.h5')

            # import pandas as pd
            hist_df = pd.DataFrame(hist.history)
            with open(f'../../models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{kfold}.csv', mode='w') as f:
                hist_df.to_csv(f)

