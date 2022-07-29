import cv2 
import tensorflow as tf
from glob import glob
import numpy as np
import os
import random
import main

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
        img = cv2.resize(img, (main.num_res, main.num_res))
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
        img = cv2.resize(img, (main.num_res, main.num_res))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        yield (img, lbl)    
            
def create_dataset(images, labels, d_type='train', aug=False):
    
    if d_type == 'test':
        return tf.data.Dataset.from_generator(test_generator, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([main.num_res, main.num_res, 3]), tf.TensorShape([1])),
                                              args=[images, labels])
        
    else:
        return tf.data.Dataset.from_generator(train_generator, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([main.num_res, main.num_res, 3]), tf.TensorShape([1])),
                                              args=[images, labels, aug])
        
def create_all_dict(dataset_path, min_num, max_num):
    all_dict = dict() 
    count_all_dict = dict() 

    for i in range(10):
        folders = os.listdir(os.path.join(dataset_path, f'H{i}'))
        
        for folder in folders:
            imgs = glob(f'{dataset_path}/H{i}/{folder}/*.jpg')
            
            folder = folder.lower().replace(' ', '')

            # class 통합 관련 내용 변경
            if folder in main.name_dict: 
                folder = main.name_dict[folder]
                
            if folder not in main.class_list:
                print(f'WARNING!! NOT FOUND LABEL : {folder}')
            
            if folder not in count_all_dict:
                count_all_dict[folder] = len(imgs) 
            else:
                count_all_dict[folder] += len(imgs)

    new_count_dict = count_all_dict.copy()


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

        
def create_train_list(dataset_path, all_dict, count_all_dict):
    images = []
    for i in range(6):
        
        folders = os.listdir(os.path.join(dataset_path, f'H{i}'))
        
        for folder in folders:
            # folder = folder.lower().replace(' ', '')
            reg_folder = folder.lower().replace(' ', '') 
            if (reg_folder in all_dict) or (reg_folder in main.name_dict):
                img = glob(dataset_path + f'/H{str(i)}/{folder}/*.jpg')
                images.extend(img)
                

        # for key, val in all_dict.items():
        #     img = glob(dataset_path + f'/H{str(i)}/{key}/*.jpg')
        #     images.extend(img)

        # for key, val in main.name_dict.items():
        #     img = glob(dataset_path + f'/H{str(i)}/{key}/*.jpg')
        #     images.extend(img)
        
    # 전남대 추가
    folders = os.listdir(os.path.join(dataset_path, 'H9'))
        
    for folder in folders:
        reg_folder = folder.lower().replace(' ', '') 
        if (reg_folder in all_dict) or (reg_folder in main.name_dict):
            img = glob(dataset_path + f'/H9/{folder}/*.jpg')
            images.extend(img)

        
        # 전남대 추가
        # for key, val in all_dict.items(): 
        #     img = glob(dataset_path + '/H9/{key}/*.jpg')
        #     images.extend(img) 

        # for key, val in main.name_dict.items():
        #     img = glob(dataset_path + f'/H9/{key}/*.jpg')
        #     images.extend(img)

    # 고른 데이터 분배를 위한 random shuffle
    random.shuffle(images)

    train_images = []
    for idx_imgs, val_imgs in enumerate(images):

        # class 통합 관련 내용 변경
        classes = val_imgs.split('/')[-2].lower().replace(' ', '')
        
        if classes in main.name_dict:
            classes = main.name_dict[classes]
        
        if count_all_dict[classes] > 0:
            count_all_dict[classes] -= 1
            train_images.append(val_imgs)
        
    train_labels = [] 
    for img in train_images:
        # lbl = img.split('/')[-1].split('\\')[0]
        lbl = img.split('/')[-2].lower().replace(' ', '')

        # 변경/통합 버전으로 label 처리
        if lbl in main.name_dict:
            lbl = main.name_dict[lbl]

        print(f'img : {img}, lbl : {lbl}')

        lbl = label_2_index(lbl, all_dict)
        train_labels.append(lbl)
        
    train_images = np.reshape(train_images, [-1, 1])
    train_labels = np.reshape(train_labels, [-1, 1])
    
    
    return train_images, train_labels