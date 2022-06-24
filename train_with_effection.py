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

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import accuracy_score
import random 
import math

import time
import pandas as pd

AUTOTUNE = tf.data.AUTOTUNE

N_BEF_RES = 256
N_RES = 256 
N_BATCH = 32 

PATH = 'C:/Users/user/Desktop/Child Skin Disease'
dataset = os.path.join(PATH, 'Total_Dataset')

# effect positive
effection = ['Abscess',
             'Cellulitis',
             'Chicken pox (varicella)',
             'Cutaneous larva migrans',
             'Eczema herpeticum',
             'Folliculitis',
             'Furuncle',
             'Green nail syndrome',
             'Herpes simplex infection',
             'Herpes zoster',
             'Impetigo',
             'Molluscum contagiosum',
             'Paronychia',
             'Staphylococcal scalded skin syndrome',
             'Tinea capitis',
             'Tinea corporis',
             'Tinea cruris',
             'Tinea faciale', 
             'Tinea manus',
             'Tinea pedis',
             'Verruca plana',
             'Viral exanthem',
             'Wart']

def cutmix(images, labels):
    # imgs = []; labs = []
    # for i in range(N_BATCH):
    APPLY = tf.cast(tf.random.uniform(()) >= 0.5, tf.int32)
    idx = tf.random.uniform((), 0, len(train_images), tf.int32)
    
    # random_img = 0 
    # random_lbl = 0 
    
    random_img = cv2.imread(train_images[idx], cv2.COLOR_BGR2YCR_CB)
    random_img = cv2.resize(random_img, (N_BEF_RES, N_BEF_RES))
    random_img = cv2.normalize(random_img, None, 0, 255, cv2.NORM_MINMAX)
    
    random_idx = train_images[idx].split('\\')[2]
    
    random_key = 0 
    if random_idx in effection:
        random_key = 1
        
    random_lbl = tf.keras.utils.to_categorical(random_key, N_CLASSES)

    W = N_RES
    H = N_RES
    lam = tf.random.uniform(())
    cut_ratio = tf.math.sqrt(1.-lam)
    cut_w = tf.cast(W * cut_ratio, tf.int32) * APPLY
    cut_h = tf.cast(H * cut_ratio, tf.int32) * APPLY

    cx = tf.random.uniform((), int(W/8), int(7/8*W), tf.int32)
    cy = tf.random.uniform((), int(H/8), int(7/8*H), tf.int32)

    xmin = tf.clip_by_value(cx - cut_w//2, 0, W)
    ymin = tf.clip_by_value(cy - cut_h//2, 0, H)
    xmax = tf.clip_by_value(cx + cut_w//2, 0, W)
    ymax = tf.clip_by_value(cy + cut_h//2, 0, H)

    mid_left = images[ymin:ymax, :xmin, :]
    # mid_mid = images[idx, ymin:ymax, xmin:xmax, :]
    mid_mid = random_img[ymin:ymax, xmin:xmax, :]
    mid_right = images[ymin:ymax, xmax:, :]
    middle = tf.concat([mid_left, mid_mid, mid_right], axis=1)
    top = images[:ymin, :, :]
    bottom = images[ymax:, :, :]
    new_img = tf.concat([top, middle, bottom], axis=0)
    # imgs.append(new_img)

    cut_w_mod = xmax - xmin
    cut_h_mod = ymax - ymin
    alpha = tf.cast((cut_w_mod*cut_h_mod)/(W*H), tf.float32)
    # label1 = labels[i]
    label1 = labels
    # label2 = labels[idx]
    label2 = random_lbl
    new_label = ((1-alpha)*label1 + alpha*label2)
    # labs.append(new_label)
        
    # new_imgs = tf.reshape(tf.stack(imgs), [-1, N_RES, N_RES, 3])
    # new_labs = tf.reshape(tf.stack(labs), [-1, N_CLASSES])

    return new_img, new_label

def train_skin_data(files):
    
    for file in files:
    
        f = file.decode('utf-8')
        
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_BEF_RES, N_BEF_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        idx = f.split('\\')[2]
        
        key = 0 
        if idx in effection:
            key = 1 
            
        lbl = tf.keras.utils.to_categorical(key, len(train_dict))

        yield (img, lbl)    
        
        # if lower than base num, should apply data augmentation
        # if base_num <= int(train_dict[idx]):
        if key == 0:
            
            # def aug(image, label):
            #     image = tf.image.random_crop(image, [RES, RES, 3])
            #     image = tf.image.random_flip_left_right(image)
            #     return image, label
            # grayscaled_img = tf.image.rgb_to_grayscale(img) 
            # yield (grayscaled_img, lbl)
            
            # saturated
            saturated_img = tf.image.adjust_saturation(img, 3)
            yield (saturated_img, lbl)
            
            
            # crop centre 
            # crop_centre_img = tf.image.central_crop(img, central_fraction=0.5)
            # yield (crop_centre_img, lbl)
            
            
            # flip 
            random_flip_img = tf.image.random_flip_left_right(img)
            yield (random_flip_img, lbl) 
            
            # Btight 
            random_bright_img = tf.image.random_brightness(img, max_delta=0.5)
            # random_bright_img = tf.clip_by_value(random_bright_img, 0, 255)
            # random_bright_image = tf.keras.preprocessing.image.array_to_img(random_bright_img)
            yield (random_bright_img, lbl) 
    
            # rotation 90 
            rotated_img = tf.image.rot90(img)        
            yield (rotated_img, lbl) 
            
            # rotation 180
            rotated_img = tf.image.rot90(img, k=2)        
            yield (rotated_img, lbl) 
            
            # rotation 270 
            rotated_img = tf.image.rot90(img, k=3)        
            yield (rotated_img, lbl) 
            
            # curmix 
            cutmixed_img, cutmixed_lbl = cutmix(img, lbl)
            yield (cutmixed_img, cutmixed_lbl)
            
            
def test_skin_data(files):
    
    for file in files:
    
        f = file.decode('utf-8')
        
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_BEF_RES, N_BEF_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # lbl = tf.keras.utils.to_categorical(label_to_index[f.split('\\')[1].split('/')[2]], len(train_dict))
        idx = f.split('\\')[2]
        
        key = 0 
        if idx in effection:
            key = 1 
            
        lbl = tf.keras.utils.to_categorical(key, len(train_dict))

        yield (img, lbl)    
        
        
def get_dropout(input_tensor, p=0.3, mc=False):
    if mc: 
        layer = Dropout(p, name='top_dropout')
        return layer(input_tensor, training=True)
    else:
        return Dropout(p, name='top_dropout')(input_tensor, training=False)
    
def create_class_weight(labels_dict, n_classes=10):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for idx, key in zip(range(N_CLASSES), keys):
        score = total / (n_classes * train_dict[key])
        class_weight[idx] = score
        
    return class_weight


def run_expriment(model_name, train_dataset, val_dataset, optimizer='adam', trainable=False, batch_size=32, mc=False, epochs=100): 
    
    if model_name == 'efficient':
        base_model = keras.applications.EfficientNetB4(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model.trainable = trainable
        
        inputs = keras.Input(shape=(N_RES, N_RES, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        x = get_dropout(x, mc)
        x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    # VGG16 
    else:
        base_model = keras.applications.VGG16(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model.trainable = True
        
        inputs = keras.Input(shape=(N_RES, N_RES, 3))
        x = base_model(inputs)
        x = keras.layers.Flatten(name = "avg_pool")(x) 
        x = keras.layers.Dense(512, activation='relu')(x)
        x = get_dropout(x, mc)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        

    sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'{PATH}/models/{model_name}_mc-{mc}_bs-{batch_size}_{time.strftime("%Y%m%d-%H%M%S")}.h5'), 
                                             monitor='val_accuracy', 
                                             verbose=0, 
                                             save_best_only=True,
                                             save_weights_only=True, 
                                             mode='max', 
                                             save_freq='epoch'), 
          tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                           patience = 4, 
                                           mode='auto',
                                           min_delta = 0.01)
          ]

    
    LR = 0.0001
    steps_per_epoch = len(train_images) // batch_size
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LR, steps_per_epoch*30, 0.1, True)
    
    # sgd = tf.keras.optimizers.SGD(0.01)
    # moving_avg_sgd = tfa.optimizers.MovingAverage(sgd)
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
    else:
        optimizer = tf.keras.optimizers.SGD(lr_schedule)
    
    model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), 
                  optimizer = optimizer,
                  metrics=['accuracy'])
    
    hist = model.fit(train_dataset,
                    validation_data = val_dataset,
                    epochs = epochs,
                    # class_weight=class_weights, 
                    verbose = 1)
    
    
    return model, hist
        
        
train_dict = {}
test_dict = {} 

for i in range(6):
    files = os.listdir(os.path.join(dataset, f'H{i}'))
    
    for f in files: 
               
        imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        # print(len(imgs))
        # print(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
        # if len(imgs) > min_num and len(imgs) <= max_num: 
        key = 0 
        if f in effection:
            key = 1
            
        if key in train_dict:
            train_dict[key] = train_dict[key] + len(imgs)
        else:
            train_dict[key] = len(imgs)
            
for i in range(7, 10): 
    # files = [val for val in list(train_dict.keys())]
    files = os.listdir(os.path.join(dataset, f'H{i}'))
    
    for f in files:
        
        imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
        key = 0 
        if f in effection:
            key = 1
        
        if key in test_dict:
            test_dict[key] = test_dict[key] + len(imgs) 
        else:
            test_dict[key] = len(imgs) 
            
N_CLASSES = len(train_dict)


train_images = [] 
test_images = []

for i in range(6):
    # for key in train_dict.keys():
    img = glob(dataset + f'/H{str(i)}/*/*.jpg')
    train_images.extend(img) 
        
for i in range(7, 10):
    # for key in train_dict.keys():
    img = glob(dataset + f'/H{str(i)}/*/*.jpg')
    test_images.extend(img) 
        
        
random.shuffle(train_images)
random.shuffle(test_images)


train_dataset = tf.data.Dataset.from_generator(train_skin_data, 
                                               output_types=(tf.float64, tf.float32), 
                                               output_shapes=(tf.TensorShape([N_RES, N_RES, 3]), tf.TensorShape([N_CLASSES])),
                                               args=[train_images])

test_dataset = tf.data.Dataset.from_generator(test_skin_data, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([N_RES, N_RES, 3]), tf.TensorShape([N_CLASSES])),
                                              args=[test_images])


split_size = int(len(train_images) * 0.2)
split_train_dataset = train_dataset.skip(split_size)
split_val_dataset = train_dataset.take(split_size)

split_train_dataset = split_train_dataset.shuffle(128).batch(32).prefetch(AUTOTUNE)
split_val_dataset = split_val_dataset.shuffle(128).batch(32).prefetch(AUTOTUNE)

model, hist = run_expriment('efficient', split_train_dataset, split_val_dataset, optimizer='adam', batch_size=32, trainable=True, epochs=25)


# evaluation
model.save(f'{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_effection.h5')

# import pandas as pd
hist_df = pd.DataFrame(hist.history)
with open(f'{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_effection_history.csv', mode='w') as f:
    hist_df.to_csv(f)

