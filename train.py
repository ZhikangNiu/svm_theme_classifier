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
from sklearn.model_selection import KFold
import time

def setup_seed(seed):
    """
    设置随机种子
    Args:
        seed: 随机数种子

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)

def write_file_to_txt(file_path):
    """
    将文件夹中的文件路径和类别写入txt文件中
    Args:
        file_path: 存放分割好的视频文件夹总路径

    Returns:

    """
    with open('train.txt','w') as f:
        for root,dirs,files in os.walk(file_path):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(root,file)
                    f.write(image_path + ' ' + root.split('/')[-2] + '\n')
    f.close()

def load_image_data(file_path,train_file_path,sample_num=100):
    """
    从整个数据集里随机采样一定数量的数据构建train_labels和train_datas
    Args:
        file_path: 存放分割好的视频文件夹总路径
        train_file_path: 训练文件train.txt的路径
        sample_num: 随机采样的数目

    Returns:
        train_datas: 训练数据
        train_labels: 训练标签

    """
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
    """
    提取图像特征
    Args:
        image_path:图片路径

    Returns:
        nparray格式的数据
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((256, 256))
    img = np.asarray(img) / 255
    img = img.flatten()
    return img

def train_svm(train_data, train_label,main_label,fold=5):
    """
    训练svm模型
    Args:
        train_data: 训练数据
        train_label: 训练标签
        main_label: 训练的主类别
        fold: kfold折数

    Returns:

    """
    classifier = svm.SVC()
    print(f"共有{len(train_label)}组数据")
    kfold = KFold(n_splits=fold, shuffle=True)
    best_acc = 0
    for train_index,test_index in kfold.split(train_data):
        # 选择训练集和测试集
        print(f"kfold={fold}，训练集：{len(train_index)}，测试集：{len(test_index)}")
        train_data_kfold = train_data[train_index]
        train_label_kfold = train_label[train_index]
        test_data_kfold = train_data[test_index]
        test_label_kfold = train_label[test_index]
        # 对标签进行转换，主要类标为1，次要类标为0
        train_label_kfold = np.where(train_label_kfold == main_label, 1, 0)
        test_label_kfold = np.where(test_label_kfold == main_label, 1, 0)
        print(f"训练集中主要类标为1的个数：{np.sum(train_label_kfold)}")
        start = time.time()
        classifier.fit(train_data_kfold, train_label_kfold)
        end = time.time()
        acc = classifier.score(test_data_kfold, test_label_kfold)
        print(f"用时{end-start}s，预测正确率：{acc * 100}%")
        if acc > best_acc:
            print(f"保存当前最好模型")
            pickle.dump(classifier, open(f'svm_model_{main_label}.pkl', 'wb'))
            best_acc = acc


def inference(image_path,main_label):
    model = pickle.load(open(f'svm_model_{main_label}.pkl', 'rb'))
    img = extract_feature(image_path)
    pred = model.predict([img])
    print(pred)



if __name__ == '__main__':
    labels = ["童趣","自然风景","萌宠","运动健身"]
    setup_seed(3407)
    if not os.path.exists('train.txt'):
        write_file_to_txt('/home/public/datasets/split_data')
    for label_name in labels:
        train_data,train_label = load_image_data('/home/public/datasets/split_data','train.txt',sample_num=1000)
        print("-"*6+f"训练类别为{label_name}的模型"+"-"*6)
        train_svm(train_data,train_label,label_name,fold=5)