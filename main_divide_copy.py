import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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

# tf.debugging.set_log_device_placement(True)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


N_RES = 300 
N_CROP_RES = 256
# N_RES = 300
N_BATCH = 8
# PATH = 'C:/Users/user/Desktop/datasets/Child Skin Disease'
PATH = 'D:/Dropbox/WORK/SNUH/Child Skin Disease'
# PATH = '../../datasets/Child Skin Disease'
dataset_path = os.path.join(PATH, 'Total_Dataset')

# Train & test set
min_num = 100
max_num = 3000 
base_num = 1000 

# name_dict = {
#     'Depressed scar' : 'Acne scar', 
#     'Acquired tufted hemangioma' : 'Acquired tufted angioma', 
#     'Cyst' : 'Epidermal cyst', 
#     'Infantile hemangioma' : 'Hemangioma',
#     'ILVEN': 'Inflammatory linear verrucous epidermal nevus'
# }

name_dict = {
    'acnescarintegrated' : 'acnescar', # add 
    'depressedscar' : 'acnescar', 
    'acquiredtuftedhemangioma' : 'acquiredtuftedangioma', 
    'acquiredtuftedhamangioma' : 'acquiredtuftedangioma', # add a and e
    'cyst' : 'epidermalcyst', 
    'cystintegrated' : 'epidermalcyst', # add
    'infantilehemangioma' : 'hemangioma',
    'hemangiomaintegrated' : 'hemangioma',
    'ilven': 'inflammatorylinearverrucousepidermalnevus'
}

class_list = [
"Abscess",
"Acanthosis nigricans",
"Acne",
"Acne neonatorum",
"Acne scar",
"Acquired bilateral nevus of Ota-like macules",
"Acquired tufted angioma",
"Actinic cheilitis",
"Actinic keratosis",
"Alopecia areata",
"Androgenetic alopecia",
"Anetoderma",
"Angioedema",
"Angiofibroma",
"Angiokeratoma",
"Angular cheilitis",
"Aplasia cutis, congenital",
"Atopic dermatitis",
"Basal Cell Carcinoma of skin",
"Beau's lines",
"Becker's nevus",
"Blue nevus",
"Bowen's disease",
"Bullous disease",
"Cafe-au-lait spot",
"Cellulitis",
"Cheilitis",
"Chicken pox (varicella)",
"Childhood granulomatous periorificial dermatitis",
"Condyloma acuminata",
"Confluent and reticulated papillomatosis",
"Congenital Hemangioma",
"Congenital melanocytic nevus",
"Congenital smooth muscle hamartoma",
"Contact dermatitis",
"Corn, Callus",
"Cutaneous larva migrans",
"Cutaneous lupus erythematosus",
"Cutis marmorata",
"Cutis marmorata telangiectatica congenita (CMTC)",
"Depressed scar",
"Dermal Melanocytic Hamartoma",
"Dermatofibroma",
"Dermatofibrosarcoma protuberans",
"Digital Mucous cyst",
"Disseminated superficial actinic porokeratosis",
"Drug eruption",
"Dyshidrotic eczema",
"Dysplastic nevus",
"Eccrine poroma",
"Eczema herpeticum",
"Epidermal cyst",
"Epidermal nevus",
"Erythema ab igne",
"Erythema annulare centrifugum",
"Erythema dyschromicum perstans",
"Erythema induratum",
"Erythema multiforme",
"Erythema nodosum",
"Exfoliative dermatitis",
"Extramammary Paget'S Disease",
"Female type baldness",
"Fixed drug eruption",
"Folliculitis",
"Fordyce's spot",
"Freckle",
"Furuncle",
"Gianotti-Crosti syndrome",
"Graft Versus Host Disease",
"Granuloma annulare",
"Green nail syndrome",
"Guttate psoriasis",
"Halo nevus",
"Hand eczema",
"Hand, foot and mouth disease",
"Hemangioma",
"Henoch-Schonlein purpura",
"Herpes simplex infection",
"Herpes zoster",
"Hidradenitis suppurativa",
"Ichthyosis",
"Idiopathic guttate hypomelanosis",
"Impetigo",
"Incontinentia pigmenti",
"Infantile hemangioma",
"Infantile seborrheic dermatitis",
"Inflammatory linear verrucous epidermal nevus",
"Ingrowing nail",
"Insect bites and stings",
"Juvenile xanthogranuloma",
"Kaposi's sarcoma",
"Keloid scar",
"Keratoacanthoma",
"Keratosis pilaris",
"Lentigo",
"Lichen amyloidosis",
"Lichen nitidus",
"Lichen planus",
"Lichen simplex chronicus",
"Lichen striatus",
"Linear scleroderma",
"Lipoma",
"Livedo reticularis",
"Livedoid vasculitis",
"Localized scleroderma",
"Lymphangioma",
"Mastocytoma",
"Melanocytic nevus",
"Melanoma",
"Melanonychia",
"Melasma",
"Miliaria",
"Milium",
"Molluscum contagiosum",
"Mongolian spot",
"Mucosal melanocytic macule",
"Mycosis fungoides",
"Necrobiosis lipoidica",
"Neurofibroma",
"Neurofibromatosis",
"Nevus anemicus",
"Nevus comedonicus",
"Nevus depigmentosus",
"nevus lipomatous superficialis",
"Nevus of Ota",
"Nevus sebaceus",
"Nevus spilus",
"Nummular eczema",
"Onychodystrophy",
"Onychogryphosis",
"Onycholysis",
"Onychomycosis",
"Oral mucocele",
"Paget's disease of skin",
"Palmoplantar keratoderma",
"Panniculitis ",
"Parapsoriasis",
"Paronychia",
"Partial unilateral lentiginosis",
"Perioral dermatitis",
"Pigmented purpuric dermatosis",
"Pilomatricoma",
"Pincer nail",
"Pitted keratolysis",
"Pityriasis alba",
"Pityriasis amiantacea",
"Pityriasis lichenoides",
"Pityriasis rosea",
"Pityriasis versicolor",
"Poikiloderma of civatte",
"Porokeratosis",
"Port-Wine stain",
"Postinflammatory hyperpigmentation",
"Postinflammatory hypopigmentation",
"Progressive macular hypomelanosis",
"Prurigo",
"Prurigo pigmentosa",
"Pseudoxanthoma elasticum",
"Psoriasis",
"Purpura",
"Pustular psoriasis",
"Pustulosis palmaris et plantaris",
"Pyoderma gangrenosum",
"Pyogenic granuloma",
"Riehl's melanosis",
"Rosacea",
"Sacrococcygeal dimple",
"Salmon patch",
"Scabies",
"Scar",
"Sebaceous hyperplasia",
"Seborrheic dermatitis",
"Seborrheic keratosis",
"Senile gluteal dermatosis",
"Senile purpura",
"Skin tag",
"Spider angioma",
"Squamous cell carcinoma of skin",
"Staphylococcal scalded skin syndrome",
"Steatocystoma multiplex",
"Striae distensae",
"Subungual hemorrhage",
"Syringoma",
"Telangiectasia",
"Tinea capitis",
"Tinea corporis",
"Tinea cruris",
"Tinea faciale",
"Tinea manus",
"Tinea pedis",
"Toxic epidermal necrolysis",
"Trichotillomania",
"Ulcer",
"Urticaria",
"Urticaria pigmentosa",
"Vascular malformation",
"Vasculitis",
"Venous lake",
"Verruca plana",
"Viral exanthem",
"Vitiligo",
"Wart",
"Xanthelasma",
"Xerotic eczema",
]


class_list = list((map(lambda x : x.lower().replace(' ', ''), class_list)))


AUTOTUNE = tf.data.AUTOTUNE

def label_2_index(lbl, label_dict):
    return label_dict[lbl]

def index_2_label(idx, label_dict):
    key = [keys for keys in label_dict if (label_dict[keys] == idx)]
    return key

def train_generator(images, labels, aug=False):
    
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_RES, N_RES))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        yield (img, lbl)    
        
        # if lower than base num, should apply data augmentation
        # if base_num <= int(train_dict[idx]):
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
        
        img = img[0].decode('utf-8')
        
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_RES, N_RES))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

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
        
def create_all_dict(dataset):
    
    count_all_dict = dict() 
    for i in range(10):
        files = os.listdir(os.path.join(dataset_path, f'H{i}'))
        
        for f in files:
            
            f = f.lower().replace(' ', '')
            
            # imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
            imgs = glob(f'{dataset_path}/H{i}/{f}/*.jpg')
            
            # print(f)

            # class 통합 관련 내용 변경
            if f in name_dict: 
                f = name_dict[f]
                
            if not f in class_list:
                print(f'WARNING!! NOT FOUND LABEL : {f}')
            
            if f not in count_all_dict:
                count_all_dict[f] = len(imgs) 
            else:
                count_all_dict[f] += len(imgs)
                
    # key_dict = list(count_all_dict.keys())[num_dict]

    new_count_all_dict = count_all_dict.copy()
    # new_count_all_dict = {
    #     'normal' : count_all_dict['normal'], 
        
    # }

    # print(new_count_dict)

    # 데이터 정제
    # for key, val in count_all_dict.items():
    #     # new_count_all_dict[key] = int(val * 0.2)
        
    #     # if val < 100:
    #     #     del new_count_all_dict[key]
    #     # else:
    #     #     new_count_all_dict[key] = int(val * 0.2)
        
    #     if val > 1000:
    #         new_count_all_dict[key] = 1000 
            
    all_dict = dict() 
    idx_num = 0 
    for key, val in new_count_all_dict.items():
        # print(idx)
        all_dict[key] = idx_num 
        idx_num += 1 
    
        
    return all_dict, new_count_all_dict

        
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
        # classes = val_imgs.split('/')[-2]
        classes = val_imgs.split('/')[-1].split('\\')[0].lower().replace(' ', '')
        
        if classes in name_dict:
            classes = name_dict[classes]
            
        if classes in count_all_dict:
            if count_all_dict[classes] > 0:
                count_all_dict[classes] -= 1
                train_images.append(val_imgs)

    train_labels = [] 
    for img in train_images:
        lbl = img.split('/')[-1].split('\\')[0].lower().replace(' ', '')
        # lbl = img.split('/')[-2]

        # 변경/통합 버전으로 label 처리
        if lbl in name_dict:
            lbl = name_dict[lbl]

        lbl = label_2_index(lbl, all_dict)
        train_labels.append(lbl)
        
    train_images = np.reshape(train_images, [-1, 1])
    train_labels = np.reshape(train_labels, [-1, 1])
    
    
    return train_images, train_labels
    

def get_dropout(input_tensor, p=0.5, mc=False):
    if mc: 
        layer = Dropout(p, name='top_dropout')
        return layer(input_tensor, training=True)
    else:
        return Dropout(p, name='top_dropout')(input_tensor, training=False)
    
    
def create_class_weight(all_dict, count_all_dict):
    
    # print(f'all_dict : {all_dict}')
    # print(f'count_all_dict : {count_all_dict}')
    
    total = np.sum(list(count_all_dict.values()))
    class_weight = dict()
    
    # for key, val in count_all_dict.items():
    #     class_weight[all_dict[key]] = val / total
    for key, val in count_all_dict.items():
        class_weight[all_dict[key]] = total / (len(all_dict) * val)

    return class_weight

# @tf.function
def get_bottleneck_model(model_name):
    if model_name == 'efficient':
        base_model = keras.applications.EfficientNetB5(include_top=False, input_shape=(256, 256, 3),  weights = 'imagenet')
        
    elif model_name == 'mobilenet':
        base_model = keras.applications.MobileNetV2(include_top=False, input_shape=(256, 256, 3),  weights = 'imagenet')
        
    else:
        base_model = keras.applications.VGG16(include_top=False, input_shape=(256, 256, 3),  weights = 'imagenet')
        
    x = base_model.layers[-1].output
    x = keras.layers.Flatten()(x)
    model = keras.Model(base_model.input, x)
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
        
    # model.summary() 
        
    return model
    
@tf.function    
def get_bottleneck_feature(model, input_imgs):
    
    print(f'type : {type(input_imgs)}, {input_imgs}')
    
    return model.predict(input_imgs, verbose=0)


def create_model(model_name, res=256, trainable=False, num_trainable=100, num_classes=10, mc=False): 
    
    data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ])

    if model_name == 'efficient':
        base_model = keras.applications.EfficientNetB3(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = trainable
        
        
        if trainable:
            if num_trainable != 0:
                for layer in base_model.layers[:num_trainable]:
                    layer.trainable = False
        
        
        inputs = keras.Input(shape=(res, res, 3))
        # x = tf.image.crop_and_resize(inputs, crop_size=(N_CROP_RES, N_CROP_RES))
        x = data_augmentation(inputs) 
        x = base_model(x)
        
        # 1
        x = keras.layers.GlobalAveragePooling2D()(x) 
        x = keras.layers.BatchNormalization()(x)
        x = get_dropout(x, mc)
        
        # 2
        # x = keras.layers.Flatten(name = "avg_pool")(x) 
        # x = keras.layers.BatchNormalization()(x)
        # x = get_dropout(x, mc)
        
        # 3 
        # x = keras.layers.Flatten(name = "avg_pool")(x) 
        # x = get_dropout(x, mc)
        
        # 4
        # x = keras.layers.Flatten(name = "avg_pool")(x) 
        # x = keras.layers.BatchNormalization()(x)
        # x = get_dropout(x, mc)
        # x = keras.layers.Dense(256, activation='relu')(x)
        
        # x = keras.layers.GlobalAveragePooling2D()(x) 
        # x = keras.layers.BatchNormalization()(x)
        # x = get_dropout(x, mc)
        
        # x = keras.layers.GlobalMaxPooling2D()(x) 
        # x = keras.layers.Flatten()(x) 
        # x = keras.layers.Dense(256, activation='relu')(x)
        
        # x = keras.layers.GlobalAveragePooling2D()(x) 
        # x = keras.layers.Dense(256, activation='sigmoid')(x)
        # x = get_dropout(x, mc)
        
        # x = keras.layers.Dense(num_classes, activation='softmax')(x)
        # model = tf.keras.Model(inputs=inputs, outputs=x)
        
        # x = keras.layers.Dense(256, activation=None)(x)
        # x = keras.layers.Lambda(lambda k: tf.math.l2_normalize(k, axis=1))(x)
        
        
        x = keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
        
        
    elif model_name == 'mobilenet':
        base_model = keras.applications.MobileNetV2(include_top=False, input_shape=(res, res, 3),  weights = 'imagenet')
        base_model.trainable = trainable
        
        if trainable:
            if num_trainable != 0:
                for layer in base_model.layers[:num_trainable]:
                    layer.trainable = False
                    
        base_model = base_model.layers[-1].output                    
        
        inputs = keras.Input(shape=(res, res, 3))
        x = preprocess_input(inputs)
        x = base_model(inputs)
        
        # 1
        # x = keras.layers.GlobalAveragePooling2D()(x) 
        # x = keras.layers.BatchNormalization()(x)
        # x = get_dropout(x, mc)
        
        # 2
        # x = keras.layers.GlobalAveragePooling2D()(x) 
        # x = keras.layers.Dense(1024, activation='relu')(x)
        # x = keras.layers.Dense(1024, activation='relu')(x)
        # x = keras.layers.Dense(512, activation='relu')(x)
        
        # 3 
        # x = keras.layers.GlobalMaxPooling2D()(x) 
        # x = keras.layers.Dense(1024, activation='relu')(x)
        # x = keras.layers.Dense(1024, activation='relu')(x)
        # x = keras.layers.Dense(512, activation='relu')(x)
        
        # 4 
        x = keras.layers.GlobalMaxPooling2D()(x) 
        x = keras.layers.BatchNormalization()(x)
        x = get_dropout(x, mc)
        
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
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
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    # model.compile(
    # loss=tfa.losses.TripletSemiHardLoss(), 
    # # loss='sparse_categorical_crossentropy',
    # optimizer=tf.keras.optimizers.Adam(1e-4),
    # # optimizer=tfa.optimizers.LazyAdam(0.001),
    # metrics=['accuracy'])
    
    model.compile(loss='binary_crossentropy', 
                optimizer = tf.keras.optimizers.SGD(0.001),
                metrics=['accuracy'])

    return model 



if __name__ == '__main__':
    
    all_dict, count_all_dict = create_all_dict(dataset_path)
    # num_classes = len(all_dict)
    
    # print(f'Number of Classes : {num_classes}')
    
    # class_weights = create_class_weight(all_dict, count_all_dict)
    
    # all_dict
    for ii in range(3):
        
        key_all_dict = list(all_dict.keys())[ii]
        
        new_all_dict = {
            'normal' : 0, 
            key_all_dict : 1 
        }
        
        new_count_all_dict = {
            'normal' : count_all_dict['normal'], 
            key_all_dict : count_all_dict[key_all_dict]
        }
        
        
        print('-------------------------------------')
        print('Create Train Datasets')
        print('-------------------------------------')

        # train_images, train_labels = create_train_list(dataset_path, new_all_dict, new_count_all_dict)
        
        # X = np.reshape(train_images, [-1, N_RES * N_RES * 3])
        # y = train_labels
        
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
        print(train_images.shape)
        
        X = np.reshape(train_images, [-1, 28 * 28 * 3])
        y = train_labels
        
        print('-------------------------------------')
        print('Begin Classifier')
        print('-------------------------------------')
        
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
        
        clf = OneVsRestClassifier(SVC()).fit(X, y)
        
        # bottleneck_model = get_bottleneck_model('efficient')

        # # for skf_num in range(3, 11):
        # for skf_num in [5, 10]:
        #     skf = StratifiedKFold(n_splits=skf_num)
            
        # #     kfold = 0 
        #     for train_idx, valid_idx in skf.split(train_images, train_labels):
                
        #         with tf.device("/device:GPU:0"):
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        #     model = create_model('efficient', 
        #                             res=N_RES,
        #                             trainable=True, 
        #                             num_trainable=-2, 
        #                             mc=False)

        # train_dataset = create_dataset(train_images[train_idx], train_labels[train_idx], aug=False).batch(N_BATCH).prefetch(AUTOTUNE)
        # valid_dataset = create_dataset(train_images[valid_idx], train_labels[valid_idx]).batch(N_BATCH).prefetch(AUTOTUNE)
            # train_dataset = create_dataset(train_images, train_labels, aug=False)
            # valid_dataset = create_dataset(train_images[valid_idx], train_labels[valid_idx]).batch(N_BATCH).prefetch(AUTOTUNE)
                
                        
            # split_len = int(len(train_images) * 0.3)
            # valid_dataset = train_dataset.take(split_len).batch(N_BATCH)
            # # valid_dataset = create_dat`aset(train_images, train_labels) 
            # train_dataset = train_dataset.skip(split_len).batch(N_BATCH)

                    # train_dataset = train_dataset.batch(N_BATCH).map(lambda x, y : (get_bottleneck_feature(bottleneck_model, x), y))
                    # valid_dataset = valid_dataset.batch(N_BATCH).map(lambda x, y : (get_bottleneck_feature(bottleneck_model, x), y))
                    # train_dataset = train_dataset.batch(N_BATCH).map(bottleneck_model)
                    # valid_dataset = valid_dataset.batch(N_BATCH).map(bottleneck_model)
            
                
                    # dir_name = os.path.join('C:/Users/user/Desktop/models/child_skin_classification/', time.strftime("%Y%m%d"))
                    
                    # if not os.path.exists(dir_name):
                    #     os.makedirs(dir_name)

        #     hist = model.fit(train_dataset,
        #             validation_data=valid_dataset,
        #             # class_weight=class_weights, 
        #             # validation_split=0.3, 
        #             epochs=50,
        #             verbose=1,
        #             shuffle=True)
        

        # model.save(f'C:/Users/user/Desktop/models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_normal_and_{key_all_dict}.h5')

    # # import pandas as pd
    # hist_df = pd.DataFrame(hist.history)
    # with open(f'C:/Users/user/Desktop/models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_100_and)500_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
    #     hist_df.to_csv(f)

    # kfold += 1

