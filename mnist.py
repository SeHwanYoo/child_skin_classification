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


N_RES = 256 
N_BATCH = 128
# PATH = 'C:/Users/user/Desktop/datasets/Child Skin Disease'
PATH = '../../datasets/Child Skin Disease'
dataset_path = os.path.join(PATH, 'Total_Dataset')

# Train & test set
# min_num = 100
# max_num = 3000 
# base_num = 1000 

# name_dict = {
#     'Depressed scar' : 'Acne scar', 
#     'Acquired tufted hemangioma' : 'Acquired tufted angioma', 
#     'Cyst' : 'Epidermal cyst', 
#     'Infantile hemangioma' : 'Hemangioma',
#     'ILVEN': 'Inflammatory linear verrucous epidermal nevus'
# }


AUTOTUNE = tf.data.AUTOTUNE

def label_2_index(lbl, label_dict):
    return label_dict[lbl]

def index_2_label(idx, label_dict):
    key = [keys for keys in label_dict if (label_dict[keys] == idx)]
    return key

def train_generator(images, labels, aug=False):
    
    # print(labels.shape)
    
    for img, lbl in zip(images, labels):
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (N_RES, N_RES))
        # print(img.shape)
        
        # lbl = cv2.resize(lbl, (N_RES, N_RES))
        
        # print(lbl.shape)
        
        yield (img, lbl)
        
    
        # img = img[0].decode('utf-8')
        # img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (N_RES, N_RES))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # yield (img, lbl)    
        
        # # if lower than base num, should apply data augmentation
        # # if base_num <= int(train_dict[idx]):
        # if aug:
        #     # Btight 
        #     random_bright_tensor = tf.image.random_brightness(img, max_delta=128)
        #     random_bright_tensor = tf.clip_by_value(random_bright_tensor, 0, 255)
        #     random_bright_image = tf.keras.preprocessing.image.array_to_img(random_bright_tensor)
        #     yield (random_bright_tensor, lbl) 
    
        #     # rotation
        #     rotated_img = tf.image.rot90(img)        
        #     yield (rotated_img, lbl) 
            
            # # curmix 
            # cutmixed_img, cutmixed_lbl = cutmix(img, lbl)
            # yield (cutmixed_img, cutmixed_lbl)
            
                
def test_generator(images, labels):
    
    for img, lbl in zip(images, labels):
        
        # img = img[0].decode('utf-8')
        # img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (N_RES, N_RES))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (N_RES, N_RES))
        # lbl = cv2.resize(lbl, (N_RES, N_RES))

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
        
# def create_all_dict(dataset, min_num, max_num):
#     all_dict = dict() 
#     count_all_dict = dict() 

#     for i in range(10):
#         files = os.listdir(os.path.join(dataset_path, f'H{i}'))
        
#         for f in files:
#             # imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
#             imgs = glob(f'{dataset_path}/H{i}/{f}/*.jpg')
            
#             # print(f)

#             # class ?????? ?????? ?????? ??????
#             if f in name_dict: 
#                 f = name_dict[f]
            
#             if f not in count_all_dict:
#                 count_all_dict[f] = len(imgs) 
#             else:
#                 count_all_dict[f] += len(imgs)

#     new_count_dict = count_all_dict.copy()

#     # print(new_count_dict)

#     # ????????? ??????
#     for key, val in count_all_dict.items():
#         if val < min_num:
#             del new_count_dict[key]

#         if val > max_num:
#             new_count_dict[key] = max_num
            

#     idx_num = 0 
#     for idx, key in new_count_dict.items():
#         # print(idx)
#         all_dict[idx] = idx_num 
#         idx_num += 1 
        
        
#     return all_dict, new_count_dict

        
def create_train_list(train_images, train_labels):
    return train_images, train_labels
    
    # images = []
    # for i in range(6):

    #     for key, val in all_dict.items():
    #         img = glob(dataset + f'/H{str(i)}/{key}/*.jpg')
    #         images.extend(img)

    #     for key, val in name_dict.items():
    #         img = glob(dataset + f'/H{str(i)}/{key}/*.jpg')
    #         images.extend(img)

        
    # # ????????? ??????
    # for key, val in all_dict.items(): 
    #     img = glob(dataset + '/H9/{key}/*.jpg')
    #     images.extend(img) 

    # for key, val in name_dict.items():
    #     img = glob(dataset + f'/H9/{key}/*.jpg')
    #     images.extend(img)

    # # ?????? ????????? ????????? ?????? random shuffle
    # random.shuffle(images)

    # # max ????????? ??????
    # # count ??? ????????? count
    # # count_all_dict = all_dict.copy() 

    # train_images = []
    # for idx_imgs, val_imgs in enumerate(images):

    #     # class ?????? ?????? ?????? ??????
    #     classes = val_imgs.split('/')[-2]
    #     if classes in name_dict:
    #         if count_all_dict[name_dict[classes]] > 0:
    #             count_all_dict[name_dict[classes]] -= 1
    #             train_images.append(val_imgs)

    #         else:
    #             continue

    #     else:
    #         if count_all_dict[classes] > 0:
    #             count_all_dict[classes] -= 1
    #             train_images.append(val_imgs)
    #         else:
    #             continue


    # train_labels = [] 
    # for img in train_images:
    #     # lbl = img.split('/')[-1].split('\\')[0]
    #     lbl = img.split('/')[-2]

    #     # ??????/?????? ???????????? label ??????
    #     if lbl in name_dict:
    #         lbl = name_dict[lbl]

    #     lbl = label_2_index(lbl, all_dict)
    #     train_labels.append(lbl)
        
    # train_images = np.reshape(train_images, [-1, 1])
    # train_labels = np.reshape(train_labels, [-1, 1])
    
    
    # return train_images, train_labels
    

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


def create_model(model_name, res=256, trainable=False, classes=10, mc=False): 

    # strategy = tf.distribute.MirroredStrategy()

    # with strategy.scope():

    if model_name == 'efficient':
        base_model = keras.applications.EfficientNetB4(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = trainable
        
        inputs = keras.Input(shape=(res, res, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        x = layers.BatchNormalization()(x)
        x = get_dropout(x, mc)
        x = keras.layers.Dense(classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    # VGG16 
    else:
        base_model = keras.applications.VGG16(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = trainable
        
        inputs = keras.Input(shape=(res, res, 3))
        x = base_model(inputs)
        x = keras.layers.Flatten(name = "avg_pool")(x) 
        x = keras.layers.Dense(512, activation='relu')(x)
        x = get_dropout(x, mc)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'])

    return model 



if __name__ == '__main__':
    
    # all_dict, count_all_dict = create_all_dict(dataset_path, min_num, max_num)
    # N_CLASSES = len(all_dict)
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
    # train_images = train_images / 255.
    
    # print(train_images.shape)
    # print(train_labels[0])
    # print(np.unique(train_labels))
    
    N_CLASSES = len(np.unique(train_labels)) 
    
    # print(N_CLASSES)
    # print(train_labels.shape)
    

    for skf_num in range(3, 11):
        skf = StratifiedKFold(n_splits=skf_num)
        
        kfold = 0 
        for train_idx, valid_idx in skf.split(train_images, train_labels):
            
            # strategy = tf.distribute.MirroredStrategy()
            # with strategy.scope():
            model = create_model('efficient', res=N_RES, classes=N_CLASSES, trainable=False, mc=False)

            # print(train_labels[train_idx].shape)


            train_dataset = create_dataset(train_images[train_idx], train_labels[train_idx], aug=False) 
            valid_dataset = create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
        
            train_dataset = train_dataset.batch(N_BATCH, drop_remainder=True).shuffle(1000).prefetch(AUTOTUNE)
            valid_dataset = valid_dataset.batch(N_BATCH, drop_remainder=True).shuffle(1000).prefetch(AUTOTUNE)

            sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'C:/Users/user/Desktop/models/child_skin_classification_infection/20220713/checkpoint_{time.strftime("%Y%m%d-%H%M%S")}_mnist_kfold_{skf_num}_{kfold}.h5'), 
                                                monitor='val_accuracy', 
                                                verbose=0, 
                                                save_best_only=True,
                                                save_weights_only=False, 
                                                mode='max', 
                                                save_freq='epoch'), 
            tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                            patience = 4, 
                                            min_delta = 0.01)]

            hist = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=50,
                    verbose=1,
                    callbacks=[sv])

            model.save(f'C:/Users/user/Desktop/models/child_skin_classification_infection/20220713/{time.strftime("%Y%m%d-%H%M%S")}_mnist_kfold_{skf_num}_{kfold}.h5')

            # import pandas as pd
            hist_df = pd.DataFrame(hist.history)
            with open(f'C:/Users/user/Desktop/models/child_skin_classification_infection/20220713/{time.strftime("%Y%m%d-%H%M%S")}_mnist_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
                hist_df.to_csv(f)

            kfold += 1

