# -*- coding: utf-8 -*-
# @Time    : 2022-09-18 14:26
# @Author  : Zhikang Niu
# @FileName: train.py
# @Software: PyCharm
from sklearn import svm
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np
import pickle
import random
from PIL import Image

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def write_file_to_txt(file_path):
    with open('train.txt','w') as f:
        for root,dirs,files in os.walk(file_path):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(root,file)
                    f.write(image_path + ' ' + root.split('/')[-2] + '\n')
    f.close()

def load_image_data(file_path,train_file_path,sample_num=100):
    train_datas = []
    train_labels = []
    random_index = np.random.randint(0, 60000, sample_num)
    with open(train_file_path,'r') as f:
        info = np.asarray(f.readlines())[random_index]
        for img_info in info:
            img_path = img_info.split(" ")[0]
            img_label = img_info.split(" ")[1].strip()
            # print(os.path.join(file_path,img_label,img_path))
            img = extract_feature(os.path.join(file_path,img_label,img_path))
            train_datas.append(img)
            train_labels.append(img_label)

    return np.asarray(train_datas),np.asarray(train_labels)

def extract_feature(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((256, 256))
    img = np.asarray(img) / 255
    img = img.flatten()
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, (256, 256))
    # img = img.flatten()
    # img = img/255
    return img

def train_svm(train_data, train_label,main_label):
    classifier = svm.SVC()
    # 将label转成0，1，主要类标为1，次要类标为0
    train_label = np.where(train_label == main_label, 1, 0)
    print(f"共有{len(train_label)}组数据")
    X_train,X_test,Y_train,Y_test = train_test_split(train_data,train_label,test_size=0.2,random_state=0)
    print(f"训练集有{len(X_train)}组数据,测试集有{len(X_test)}组数据")
    classifier.fit(X_train,Y_train)
    pred = classifier.predict(X_test)
    print(f"预测正确率：{classifier.score(X_test, Y_test)*100}%")
    pickle.dump(classifier, open('svm_model_pet.pkl', 'wb'))

def inference(image_path):
    model = pickle.load(open('svm_model.pkl', 'rb'))
    img = extract_feature(image_path)
    pred = model.predict([img])
    print(pred)



if __name__ == '__main__':
    setup_seed(3407)
    if not os.path.exists('train.txt'):
        write_file_to_txt('data/train')
    train_data,train_label = load_image_data('/home/public/datasets/split_data','train.txt')
    train_svm(train_data,train_label,'萌宠')