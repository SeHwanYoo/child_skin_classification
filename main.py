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


num_res = 300
num_batch = 128
# PATH = 'C:/Users/user/Desktop/datasets/Child Skin Disease'
base_path = '../../datasets/Child Skin Disease'
dataset_path = os.path.join(base_path, 'Total_Dataset')

# Train & test set
min_num = 100
max_num = 3000 
base_num = 1000 


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

    
if __name__ == '__main__':
    
    all_dict, count_all_dict = dataset_generator.create_all_dict(min_num, max_num)
    num_classes = len(all_dict)
    
    print(f'number of classes : {num_classes}')

    train_images, train_labels = dataset_generator.create_train_list(all_dict, count_all_dict)

    # for skf_num in range(3, 11):
    for skf_num in [5, 10]:
        skf = StratifiedKFold(n_splits=skf_num)
        
        kfold = 0 
        for train_idx, valid_idx in skf.split(train_images, train_labels):
            
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = models.create_model('efficient', 
                                            res=num_res, 
                                            num_classes=num_classes, 
                                            trainable=True, 
                                            num_trainable=-2, 
                                            mc=False)


                train_dataset = dataset_generator.create_dataset(train_images[train_idx], train_labels[train_idx], aug=False) 
                valid_dataset = dataset_generator.create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
            
                train_dataset = train_dataset.batch(num_batch, drop_remainder=True).shuffle(1000).prefetch(AUTOTUNE)
                valid_dataset = valid_dataset.batch(num_batch, drop_remainder=True).shuffle(1000).prefetch(AUTOTUNE)
                

                sv = [tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(f'../../models/child_skin_classification/checkpoint_{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.h5'),monitor='val_accuracy', 
                    verbose=0, 
                    save_best_only=True,
                    save_weights_only=False, 
                    mode='max', 
                    save_freq='epoch')]
                # tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                #                                 patience = 4, 
                #                                 min_delta = 0.01)]
                
                # tensorboard = TensorBoard(log_dir=f'../../logs/child_skin_classification/{time.strftime("%Y%m%d")}_{kfold}')

                hist = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        epochs=10000,
                        # verbose=2,
                        shuffle=True)
            

            model.save(f'../../models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.h5')

            # import pandas as pd
            hist_df = pd.DataFrame(hist.history)
            with open(f'../../models/child_skin_classification/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
                hist_df.to_csv(f)

            kfold += 1

