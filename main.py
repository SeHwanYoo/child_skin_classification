import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.callbacks import TensorBoard

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

# defined model
import dataset_generator
import models

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


num_res = 256 
# N_RES = 300
num_batch = 64
# PATH = 'C:/Users/user/Desktop/datasets/Child Skin Disease'
base_path = '../../datasets/Child Skin Disease'
dataset_path = os.path.join(base_path, 'Total_Dataset')

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


    
if __name__ == '__main__':
    
    all_dict, count_all_dict = dataset_generator.create_all_dict(dataset_path, min_num, max_num)
    num_classes = len(all_dict)
    
    # print(f'number of classes : {num_classes}')

    train_images, train_labels = dataset_generator.create_train_list(dataset_path, all_dict, count_all_dict)

    # for skf_num in range(3, 11):
    for skf_num in [5, 10]:
        skf = StratifiedKFold(n_splits=skf_num)
        
        kfold = 0 
        for train_idx, valid_idx in skf.split(train_images, train_labels):
            
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = models.create_model('efficient', res=num_res, num_classes=num_classes, trainable=True, num_trainable=-2, mc=False)


            train_dataset = models.create_dataset(train_images[train_idx], train_labels[train_idx], aug=False) 
            valid_dataset = models.create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
        
            train_dataset = train_dataset.batch(num_batch, drop_remainder=True).shuffle(1000).prefetch(AUTOTUNE)
            valid_dataset = valid_dataset.batch(num_batch, drop_remainder=True).shuffle(1000).prefetch(AUTOTUNE)
            
            dir_name = os.path.join('../../models/child_skin_classification/', time.strftime("%Y%m%d"))
            
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            # sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'../../models/child_skin_classification/{dir_name}/checkpoint_{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.h5'), 
            #                                     monitor='val_accuracy', 
            #                                     verbose=0, 
            #                                     save_best_only=True,
            #                                     save_weights_only=False, 
            #                                     mode='max', 
            #                                     save_freq='epoch'), 
            # tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
            #                                 patience = 4, 
            #                                 min_delta = 0.01)]
            
            # tensorboard = TensorBoard(log_dir=f'../../logs/child_skin_classification/{time.strftime("%Y%m%d")}_{kfold}')

            hist = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=100,
                    # verbose=2,
                    shuffle=True)
            

            model.save(f'../../models/child_skin_classification/{dir_name}/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.h5')

            # import pandas as pd
            hist_df = pd.DataFrame(hist.history)
            with open(f'../../models/child_skin_classification/{dir_name}/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
                hist_df.to_csv(f)

            kfold += 1

