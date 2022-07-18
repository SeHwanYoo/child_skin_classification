from grpc import xds_server_credentials
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator

import warnings 
warnings.filterwarnings(action='ignore')

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


N_RES = 300 
N_BATCH = 32
PATH = 'C:/Users/user/Desktop/datasets/Child Skin Disease'
# PATH = '../../datasets/Child Skin Disease'
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    count_all_dict = dict() 

    for i in range(10):
        files = os.listdir(os.path.join(dataset_path, f'H{i}'))
        
        for f in files:
            # imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
            imgs = glob(f'{dataset_path}/H{i}/{f}/*.jpg')
            
            # print(f)

            # class 통합 관련 내용 변경
            if f in name_dict: 
                f = name_dict[f]
            
            if f not in count_all_dict:
                count_all_dict[f] = len(imgs) 
            else:
                count_all_dict[f] += len(imgs)

    new_count_dict = count_all_dict.copy()

    # print(new_count_dict)

    # 데이터 정제
    for key, val in count_all_dict.items():
        if val < min_num:
            del new_count_dict[key]

        if val > max_num:
            new_count_dict[key] = max_num
            

    idx_num = 0 
    for key, val in new_count_dict.items():
        # print(idx)
        all_dict[key] = idx_num 
        idx_num += 1 
        
        
    return all_dict, new_count_dict

        
def create_train_list(dataset, all_dict, count_all_dict):
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
    # count_all_dict = all_dict.copy() 

    train_images = []
    for idx_imgs, val_imgs in enumerate(images):

        # class 통합 관련 내용 변경
        # print(val_imgs)
        classes = val_imgs.split('/')[-1].split('\\')[0]
        # classes = val_imgs.split('/')[-2]
        
        
        if classes in name_dict:
            if count_all_dict[name_dict[classes]] > 0:
                count_all_dict[name_dict[classes]] -= 1
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

        lbl = img.split('/')[-1].split('\\')[0]
        # lbl = img.split('/')[-2]

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


# def run_expriment(model_name, train_dataset, val_dataset, kfold=0, res=256, classes=10, batch_size=32, mc=False, epochs=100): 

#     # strategy = tf.distribute.MirroredStrategy()

#     # with strategy.scope():

#     if model_name == 'efficient':
#         base_model = keras.applications.EfficientNetB4(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
#         base_model.trainable = True
        
#         inputs = keras.Input(shape=(res, res, 3))
#         x = base_model(inputs)
#         x = keras.layers.GlobalAveragePooling2D()(x) 
#         x = get_dropout(x, mc)
#         x = keras.layers.Dense(classes, activation='softmax')(x)
#         model = tf.keras.Model(inputs=inputs, outputs=x)
        
#     # VGG16 
#     else:
#         base_model = keras.applications.VGG16(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
#         base_model.trainable = True
        
#         inputs = keras.Input(shape=(res, res, 3))
#         x = base_model(inputs)
#         x = keras.layers.Flatten(name = "avg_pool")(x) 
#         x = keras.layers.Dense(512, activation='relu')(x)
#         x = get_dropout(x, mc)
#         x = keras.layers.Dense(256, activation='relu')(x)
#         x = keras.layers.Dense(classes, activation='softmax')(x)
#         model = tf.keras.Model(inputs=inputs, outputs=x)

#     model.compile(loss='sparse_categorical_crossentropy', 
#                 optimizer = tf.keras.optimizers.Adam(0.001), 
#                 metrics=['accuracy'])

#     return model 


def create_model(model_name, res=256, trainable=False, num_trainable=100, num_classes=10, mc=False): 

    # strategy = tf.distribute.MirroredStrategy()

    # with strategy.scope():

    if model_name == 'efficient':
        base_model = keras.applications.EfficientNetB4(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = trainable
        
        if trainable:
            for layer in base_model.layers[:num_trainable]:
                layer.trainable = False
        
        # print("Number of layers in the base model: ", len(base_model.layers))
        
        inputs = keras.Input(shape=(res, res, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        
        # add 20220714
        x = keras.layers.BatchNormalization()(x)
        
        x = get_dropout(x, mc)
        
        # add 20220714
        # x = keras.layers.Dense(512, activation='relu')(x)
        # x = keras.layers.Dense(256, activation='relu')(x)
        
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    elif model_name == 'resnet':
        # tf.keras.applications.resnet.ResNet152
        base_model = tf.keras.applications.resnet.ResNet152(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = trainable
        
        if trainable:
            for layer in base_model.layers[:num_trainable]:
                layer.trainable = False
        # base_model.layers.trainable[:-1] = True
        
        # print("Number of layers in the base model: ", len(base_model.layers))
        
        inputs = keras.Input(shape=(res, res, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        
        # add 20220714
        x = keras.layers.BatchNormalization()(x)
        
        x = get_dropout(x, mc)
        
        # add 20220714
        # x = keras.layers.Dense(512, activation='relu')(x)
        # x = keras.layers.Dense(256, activation='relu')(x)
        
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)        
        
    # VGG16 
    elif model_name == 'vgg':
        base_model = keras.applications.VGG16(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = trainable
        
        inputs = keras.Input(shape=(res, res, 3))
        x = base_model(inputs)
        x = keras.layers.Flatten(name = "avg_pool")(x) 
        x = keras.layers.Dense(512, activation='relu')(x)
        x = get_dropout(x, mc)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    else:
        inputs = Input((N_RES, N_RES, 3))

        x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs) 
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs) 
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x) 
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(128, (3, 3), padding='same')(x) 
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(128, (3, 3), padding='same')(x) 
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x) 
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Flatten()(x) 
        x = keras.layers.Dense(512)(x) 
        x = keras.layers.Activation('relu')(x) 
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Dense(10)(x) 
        x = keras.layers.Activation('softmax')(x) 

        model = keras.Model(inputs=inputs, outputs=x)
        
    # if op

    model.compile(loss='sparse_categorical_crossentropy',
    # optimizer=tf.keras.optimizers.Adam(1e-2),
    # optimizer='RMSprop', 
    # optimizer=tfa.optimizers.LazyAdam(0.001),
    optimizer=tf.keras.optimizers.SGD(1e-2),
    metrics=['accuracy'])

    return model 



if __name__ == '__main__':
    
    all_dict, count_all_dict = create_all_dict(dataset_path, min_num, max_num)
    N_CLASSES = len(all_dict)

    train_images, train_labels = create_train_list(dataset_path, all_dict, count_all_dict)

    # for skf_num in range(3, 11):
    for skf_num in [5, 10]:
        skf = StratifiedKFold(n_splits=skf_num)
    
    # datagen = ImageDataGenerator(rotation_range=10,brightness_range=[0.2,1.0])  
        kfold = 0 
        for train_idx, valid_idx in skf.split(train_images, train_labels):
            
            # strategy = tf.distribute.MirroredStrategy()
            # with strategy.scope():
            model = create_model('small', res=N_RES, num_classes=N_CLASSES, trainable=True, num_trainable=-2, mc=False)
            
            # datagen = BalancedDataGenerator() 
            
    # train_ds = BalancedDataGenerator(datagen, train_images, train_labels, batch_size=N_BATCH)
    
    # print(train_ds) 


            train_dataset = create_dataset(train_images[train_idx], train_labels[train_idx], aug=False) 
            valid_dataset = create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
            
            # balanced_gen = BalancedDataGenerator(X_train, y_train, datagen, batch_size=64)
# balanced_gen_val = BalancedDataGenerator(X_val, y_val, datagen, batch_size=64)
# steps_per_epoch = balanced_gen.steps_per_epoch
            
            # train_dataset, steps_per_epoch = train_dataset.map(test_map) 
            # self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)

        
            train_dataset = train_dataset.batch(N_BATCH, drop_remainder=True).shuffle(1000).prefetch(AUTOTUNE)
            valid_dataset = valid_dataset.batch(N_BATCH, drop_remainder=True).shuffle(1000).prefetch(AUTOTUNE)

            # sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'../../models/child_skin_classification/checkpoint_{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.h5'), 
            #                                     monitor='val_accuracy', 
            #                                     verbose=0, 
            #                                     save_best_only=True,
            #                                     save_weights_only=False, 
            #                                     mode='max', 
            #                                     save_freq='epoch'), 
            # tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
            #                                 patience = 4, 
            #                                 min_delta = 0.01)]

            hist = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=50,
                    shuffle=True, 
                    verbose=1)

            # model.save(f'../../models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.h5')

            # # import pandas as pd
            # hist_df = pd.DataFrame(hist.history)
            # with open(f'../../models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
            #     hist_df.to_csv(f)

            # kfold += 1

