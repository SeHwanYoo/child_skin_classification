<<<<<<< HEAD
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
import time
import math

AUTOTUNE = tf.data.AUTOTUNE


N_BEF_RES = 256
N_RES = 256 
# N_CLASSES = 143 
N_BATCH = 32 
PATH = 'C:/Users/user/Desktop/Child Skin Disease'
dataset = os.path.join(PATH, 'Total_Dataset')

# Train & test set
min_num = 100
max_num = 3000 
base_num = 1000 # baseline of data augmentation

train_dict = {}
test_dict = {} 

N_CLASSES = 0

train_images = [] 
test_images = []

def train_skin_data(files):
    
    for file in files:
    
        f = file.decode('utf-8')
        
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_BEF_RES, N_BEF_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        idx = f.split('\\')[1].split('/')[2]
        lbl = tf.keras.utils.to_categorical(label_to_index[idx], len(train_dict))

        yield (img, lbl)    
        
        # if lower than base num, should apply data augmentation
        if base_num <= int(train_dict[idx]):

            # Btight 
            random_bright_tensor = tf.image.random_brightness(img, max_delta=128)
            random_bright_tensor = tf.clip_by_value(random_bright_tensor, 0, 255)
            random_bright_image = tf.keras.preprocessing.image.array_to_img(random_bright_tensor)
            yield (random_bright_tensor, lbl) 
    
            # rotation
            rotated_img = tf.image.rot90(img)        
            yield (rotated_img, lbl) 
            
            # curmix 
            cutmixed_img, cutmixed_lbl = cutmix(img, lbl)
            yield (cutmixed_img, cutmixed_lbl)
            
        # else:
        #     random_apply = tf.random.uniform(()) >= 0.3
            
        #     if random_apply:
                
            
def test_skin_data(files):
    
    for file in files:
    
        f = file.decode('utf-8')
        a
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_BEF_RES, N_BEF_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        lbl = tf.keras.utils.to_categorical(label_to_index[f.split('\\')[1].split('/')[2]], len(train_dict))

        yield (img, lbl)           
        
        
def cutmix(images, labels):
    # imgs = []; labs = []
    # for i in range(N_BATCH):
    APPLY = tf.cast(tf.random.uniform(()) >= 0.5, tf.int32)
    idx = tf.random.uniform((), 0, len(train_images), tf.int32)
    tf.random.uniform((), 0, 100, tf.int32)
    
    # random_img = 0 
    # random_lbl = 0 
    
    random_img = cv2.imread(train_images[idx], cv2.COLOR_BGR2YCR_CB)
    random_img = cv2.resize(random_img, (N_BEF_RES, N_BEF_RES))
    random_img = cv2.normalize(random_img, None, 0, 255, cv2.NORM_MINMAX)
    random_lbl = tf.keras.utils.to_categorical(label_to_index[train_images[idx].split('\\')[1].split('/')[2]], len(train_dict))

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

def get_dropout(input_tensor, p=0.3, mc=False):
    if mc: 
        layer = Dropout(p, name='top_dropout')
        return layer(input_tensor, training=True)
    else:
        return Dropout(p, name='top_dropout')(input_tensor, training=False)
    
    
# def create_class_weight(labels_dict, mu=0.15):
#     total = np.sum(list(labels_dict.values()))
#     keys = labels_dict.keys()
#     class_weight = dict()
    
#     for idx, key in zip(range(N_CLASSES), keys):
#         score = math.log(mu*total/float(labels_dict[key]))
#         class_weight[idx] = score
    
#     return class_weight

def create_class_weight(labels_dict, n_classes=10):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for idx, key in zip(range(N_CLASSES), keys):
        score = total / (n_classes * train_dict[key])
        class_weight[idx] = score
        
    return class_weight

def run_expriment(model_name, train_dataset, val_dataset, batch_size=32, mc=False, epochs=100, n_splits=5): 
    
    if model_name == 'efficient':
        # base_model = keras.applications.EfficientNetB0(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model = keras.applications.EfficientNetB7(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model.trainable = True
        
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
        

    sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'C:/workspace/child_skin_classificiation/models/{model_name}_mc-{str(mc)}_bs-{batch_size}_{time.strftime("%Y%m%d-%H%M%S")}.h5'), 
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
    steps_per_epoch = len(train_images) // batch_size
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LR, steps_per_epoch*30, 0.1, True)
    
    sgd = tf.keras.optimizers.SGD(0.01)
    moving_avg_sgd = tfa.optimizers.MovingAverage(sgd)
    
    model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), 
                  optimizer = moving_avg_sgd, 
                #   optimizer = tf.keras.optimizers.Adam(lr_schedule), 
                #   optimizer = tf.keras.optimizers.Adam(lr_schedule), 
                  metrics=['accuracy'])
    
    hist = model.fit(train_dataset,
                    validation_data = val_dataset,
                    epochs = epochs,
                    class_weight=class_weights, 
                    verbose = 1,
                    shuffle = True,
                    callbacks = [sv])
    
    # histories.append(hist)
    
    return model, hist

for i in range(6):
    files = os.listdir(os.path.join(dataset, f'H{i}'))
    
    for f in files: 
        imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
        if len(imgs) > min_num and len(imgs) <= max_num: 
            
            if f in train_dict:
                train_dict[f] = train_dict[f] + len(imgs)
            else:
                train_dict[f] = len(imgs)
            
for i in range(7, 10): 
    files = [val for val in list(train_dict.keys())]
    
    for f in files:
        imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
        if f in test_dict:
            test_dict[f] = test_dict[f] + len(imgs) 
        else:
            test_dict[f] = len(imgs) 

N_CLASSES = len(train_dict)
class_weights = create_class_weight(train_dict, N_CLASSES)

for i in range(6):
    for key in train_dict.keys():
        img = glob(dataset + f'/H{str(i)}/{key}/*.jpg')
        train_images.extend(img) 
        
for i in range(7, 10):
    for key in train_dict.keys():
        img = glob(dataset + f'/H{str(i)}/{key}/*.jpg')
        test_images.extend(img) 
        
        
random.shuffle(train_images)
random.shuffle(test_images)


label_to_index = {}
for idx, key in zip(range(len(train_dict)), train_dict.keys()):
    label_to_index[key] = idx



train_dataset = tf.data.Dataset.from_generator(train_skin_data, 
                                               output_types=(tf.float64, tf.float32), 
                                               output_shapes=(tf.TensorShape([N_BEF_RES, N_BEF_RES, 3]), tf.TensorShape([N_CLASSES])),
                                               args=[train_images])

test_dataset = tf.data.Dataset.from_generator(test_skin_data, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([N_BEF_RES, N_BEF_RES, 3]), tf.TensorShape([N_CLASSES])),
                                              args=[test_images])


split_size = int(len(train_images) * 0.2)
split_train_dataset = train_dataset.skip(split_size)
split_val_dataset = train_dataset.take(split_size)

# new_split_train_dataset = split_train_dataset.map(aug, num_parallel_calls=AUTOTUNE).batch(N_BATCH, drop_remainder=True).prefetch(AUTOTUNE)
# val_dataset = val_dataset.map(aug, num_parallel_calls=AUTOTUNE).batch(N_BATCH, drop_remainder=True).prefetch(AUTOTUNE)

# batch_list = [1, 8, 64]
# batch_list = [16, 32, 64]
batch_list = [64]

for batch in batch_list:
    
    # new_split_train_dataset = split_train_dataset.map(aug, num_parallel_calls=AUTOTUNE).batch(batch, drop_remainder=True).prefetch(AUTOTUNE)
    # new_split_vbal_dataset = split_val_dataset.map(aug, num_parallel_calls=AUTOTUNE).batch(batch, drop_remainder=True).prefetch(AUTOTUNE)
    new_split_train_dataset = split_train_dataset.shuffle(150).batch(batch, drop_remainder=True).prefetch(AUTOTUNE)
    new_split_val_dataset = split_val_dataset.shuffle(150).batch(batch, drop_remainder=True).prefetch(AUTOTUNE)

    with tf.device('/device:GPU:0'):
=======
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
import time
import math

AUTOTUNE = tf.data.AUTOTUNE


N_BEF_RES = 256
N_RES = 256 
# N_CLASSES = 143 
N_BATCH = 32 
PATH = 'C:/Users/user/Desktop/Child Skin Disease'
dataset = os.path.join(PATH, 'Total_Dataset')

# Train & test set
min_num = 100
max_num = 3000 
base_num = 1000 # baseline of data augmentation

train_dict = {}
test_dict = {} 

N_CLASSES = 0

train_images = [] 
test_images = []

def train_skin_data(files):
    
    for file in files:
    
        f = file.decode('utf-8')
        
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_BEF_RES, N_BEF_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        idx = f.split('\\')[1].split('/')[2]
        lbl = tf.keras.utils.to_categorical(label_to_index[idx], len(train_dict))

        yield (img, lbl)    
        
        # if lower than base num, should apply data augmentation
        if base_num <= int(train_dict[idx]):

            # Btight 
            random_bright_tensor = tf.image.random_brightness(img, max_delta=128)
            random_bright_tensor = tf.clip_by_value(random_bright_tensor, 0, 255)
            random_bright_image = tf.keras.preprocessing.image.array_to_img(random_bright_tensor)
            yield (random_bright_tensor, lbl) 
    
            # rotation
            rotated_img = tf.image.rot90(img)        
            yield (rotated_img, lbl) 
            
            # curmix 
            cutmixed_img, cutmixed_lbl = cutmix(img, lbl)
            yield (cutmixed_img, cutmixed_lbl)
            
        # else:
        #     random_apply = tf.random.uniform(()) >= 0.3
            
        #     if random_apply:
                
            
def test_skin_data(files):
    
    for file in files:
    
        f = file.decode('utf-8')
        a
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_BEF_RES, N_BEF_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        lbl = tf.keras.utils.to_categorical(label_to_index[f.split('\\')[1].split('/')[2]], len(train_dict))

        yield (img, lbl)           
        
        
def cutmix(images, labels):
    # imgs = []; labs = []
    # for i in range(N_BATCH):
    APPLY = tf.cast(tf.random.uniform(()) >= 0.5, tf.int32)
    idx = tf.random.uniform((), 0, len(train_images), tf.int32)
    tf.random.uniform((), 0, 100, tf.int32)
    
    # random_img = 0 
    # random_lbl = 0 
    
    random_img = cv2.imread(train_images[idx], cv2.COLOR_BGR2YCR_CB)
    random_img = cv2.resize(random_img, (N_BEF_RES, N_BEF_RES))
    random_img = cv2.normalize(random_img, None, 0, 255, cv2.NORM_MINMAX)
    random_lbl = tf.keras.utils.to_categorical(label_to_index[train_images[idx].split('\\')[1].split('/')[2]], len(train_dict))

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

def get_dropout(input_tensor, p=0.3, mc=False):
    if mc: 
        layer = Dropout(p, name='top_dropout')
        return layer(input_tensor, training=True)
    else:
        return Dropout(p, name='top_dropout')(input_tensor, training=False)
    
    
# def create_class_weight(labels_dict, mu=0.15):
#     total = np.sum(list(labels_dict.values()))
#     keys = labels_dict.keys()
#     class_weight = dict()
    
#     for idx, key in zip(range(N_CLASSES), keys):
#         score = math.log(mu*total/float(labels_dict[key]))
#         class_weight[idx] = score
    
#     return class_weight

def create_class_weight(labels_dict, n_classes=10):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for idx, key in zip(range(N_CLASSES), keys):
        score = total / (n_classes * train_dict[key])
        class_weight[idx] = score
        
    return class_weight

def run_expriment(model_name, train_dataset, val_dataset, batch_size=32, mc=False, epochs=100, n_splits=5): 
    
    if model_name == 'efficient':
        # base_model = keras.applications.EfficientNetB0(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model = keras.applications.EfficientNetB7(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model.trainable = True
        
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
        

    sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'C:/workspace/child_skin_classificiation/models/{model_name}_mc-{str(mc)}_bs-{batch_size}_{time.strftime("%Y%m%d-%H%M%S")}.h5'), 
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
    steps_per_epoch = len(train_images) // batch_size
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LR, steps_per_epoch*30, 0.1, True)
    
    sgd = tf.keras.optimizers.SGD(0.01)
    moving_avg_sgd = tfa.optimizers.MovingAverage(sgd)
    
    model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), 
                  optimizer = moving_avg_sgd, 
                #   optimizer = tf.keras.optimizers.Adam(lr_schedule), 
                #   optimizer = tf.keras.optimizers.Adam(lr_schedule), 
                  metrics=['accuracy'])
    
    hist = model.fit(train_dataset,
                    validation_data = val_dataset,
                    epochs = epochs,
                    class_weight=class_weights, 
                    verbose = 1,
                    shuffle = True,
                    callbacks = [sv])
    
    # histories.append(hist)
    
    return model, hist

for i in range(6):
    files = os.listdir(os.path.join(dataset, f'H{i}'))
    
    for f in files: 
        imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
        if len(imgs) > min_num and len(imgs) <= max_num: 
            
            if f in train_dict:
                train_dict[f] = train_dict[f] + len(imgs)
            else:
                train_dict[f] = len(imgs)
            
for i in range(7, 10): 
    files = [val for val in list(train_dict.keys())]
    
    for f in files:
        imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
        if f in test_dict:
            test_dict[f] = test_dict[f] + len(imgs) 
        else:
            test_dict[f] = len(imgs) 

N_CLASSES = len(train_dict)
class_weights = create_class_weight(train_dict, N_CLASSES)

for i in range(6):
    for key in train_dict.keys():
        img = glob(dataset + f'/H{str(i)}/{key}/*.jpg')
        train_images.extend(img) 
        
for i in range(7, 10):
    for key in train_dict.keys():
        img = glob(dataset + f'/H{str(i)}/{key}/*.jpg')
        test_images.extend(img) 
        
        
random.shuffle(train_images)
random.shuffle(test_images)


label_to_index = {}
for idx, key in zip(range(len(train_dict)), train_dict.keys()):
    label_to_index[key] = idx



train_dataset = tf.data.Dataset.from_generator(train_skin_data, 
                                               output_types=(tf.float64, tf.float32), 
                                               output_shapes=(tf.TensorShape([N_BEF_RES, N_BEF_RES, 3]), tf.TensorShape([N_CLASSES])),
                                               args=[train_images])

test_dataset = tf.data.Dataset.from_generator(test_skin_data, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([N_BEF_RES, N_BEF_RES, 3]), tf.TensorShape([N_CLASSES])),
                                              args=[test_images])


split_size = int(len(train_images) * 0.2)
split_train_dataset = train_dataset.skip(split_size)
split_val_dataset = train_dataset.take(split_size)

# new_split_train_dataset = split_train_dataset.map(aug, num_parallel_calls=AUTOTUNE).batch(N_BATCH, drop_remainder=True).prefetch(AUTOTUNE)
# val_dataset = val_dataset.map(aug, num_parallel_calls=AUTOTUNE).batch(N_BATCH, drop_remainder=True).prefetch(AUTOTUNE)

# batch_list = [1, 8, 64]
# batch_list = [16, 32, 64]
batch_list = [64]

for batch in batch_list:
    
    # new_split_train_dataset = split_train_dataset.map(aug, num_parallel_calls=AUTOTUNE).batch(batch, drop_remainder=True).prefetch(AUTOTUNE)
    # new_split_vbal_dataset = split_val_dataset.map(aug, num_parallel_calls=AUTOTUNE).batch(batch, drop_remainder=True).prefetch(AUTOTUNE)
    new_split_train_dataset = split_train_dataset.shuffle(150).batch(batch, drop_remainder=True).prefetch(AUTOTUNE)
    new_split_val_dataset = split_val_dataset.shuffle(150).batch(batch, drop_remainder=True).prefetch(AUTOTUNE)

    with tf.device('/device:GPU:0'):
>>>>>>> 661fc3f1ec1732ee782fcdd19f4ccfb4801a8724
        model, hist = run_expriment('efficient', new_split_train_dataset, new_split_val_dataset, batch_size=batch, mc=True, epochs=100)