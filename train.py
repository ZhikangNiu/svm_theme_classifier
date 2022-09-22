# -*- coding: utf-8 -*-
# @Time    : 2022-09-18 14:26
# @Author  : Zhikang Niu
# @FileName: train.py
# @Software: PyCharm
import glob
from math import ceil

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
from sklearn.decomposition import PCA
import torch
from torchvision.models import resnet18,resnet50
from sklearn.model_selection import GridSearchCV
def setup_seed(seed):
    """
    设置随机种子
    Args:
        seed: 随机数种子

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)

def write_train_file_to_txt(file_path):
    """
    将文件夹中的训练集文件路径和类别写入txt文件中
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

def generate_test_index(every_label_nums=50,use_cnn=False,model=None,use_pca=True):
    label_1_test_index=[x for x in range(0,16412,ceil(16412/every_label_nums))]
    label_2_test_index=[x for x in range(16413,39752,ceil((39752-16413)/every_label_nums))]
    label_3_test_index=[x for x in range(39753,43905,ceil((43905-39753)/every_label_nums))]
    label_4_test_index = [x for x in range(43907,60000,ceil((60000-43907)/every_label_nums))]
    test_index = [*label_1_test_index,*label_2_test_index,*label_3_test_index,*label_4_test_index]
    test_datas = []
    test_labels = []
    with open('train.txt','r') as f:
        info = np.asarray(f.readlines())[test_index]
        for img_info in info:
            img_path = img_info.split(" ")[0]
            img_label = img_info.split(" ")[1].strip()
            try:
                img = extract_feature(img_path,use_cnn=use_cnn,model=model,use_pca=use_pca)
                test_datas.append(img)
                test_labels.append(img_label)
            except None:
                continue
    return np.asarray(test_datas),np.asarray(test_labels)


    # return test_info
    # for label in labels:



def load_image_data(file_path,train_file_path=None,mode='train',sample_num=100,use_cnn=False,model=None,use_pca=True):

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
    if mode == 'train' or mode == 'test':
        datas = []
        labels = []
        random_index = np.random.randint(0, 60000, sample_num)
        with open(train_file_path,'r') as f:
            info = np.asarray(f.readlines())[random_index]
            for img_info in info:
                img_path = img_info.split(" ")[0]
                img_label = img_info.split(" ")[1].strip()
                # print(os.path.join(file_path,img_label,img_path))
                img = extract_feature(os.path.join(file_path,img_label,img_path),use_cnn=use_cnn,model=model,use_pca=use_pca)
                datas.append(img)
                labels.append(img_label)

        return np.asarray(datas),np.asarray(labels)
    if mode == 'infer':
        datas = []
        # 获取一个视频下所有的图片
        datas_path = glob.glob(os.path.join(file_path,'*.jpg'))
        for data_path in datas_path:
            img = extract_feature(data_path,use_cnn=use_cnn,model=model,use_pca=use_pca)
            datas.append(img)
        return np.asarray(datas)

def extract_feature(image_path,use_cnn=False,model=None,use_pca=True):
    """
    提取图像特征
    Args:
        image_path:图片路径

    Returns:
        nparray格式的数据
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    if use_cnn:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 255
        output = model(img)
        img = output.squeeze(0).detach().numpy()
        # img = img.flatten()
        img = img.reshape(img.shape[0], -1)
    else:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if use_pca:
        pca = PCA(n_components=20)
        img_pca = pca.fit_transform(img)
    img = img_pca.flatten()
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
    #TODO:使用不同的核函数训练svm模型
    ker_list = ['linear', 'poly', 'rbf', 'sigmoid']
    for ker in ker_list:
        best_acc = 0
        print("svm use kernel:{}".format(ker))
        #TODO:使用GridSearchCV寻找最优参数，懒了
        classifier = svm.SVC(kernel=ker)

        print(f"共有{len(train_label)}组数据")
        kfold = KFold(n_splits=fold, shuffle=True)
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
                pickle.dump(classifier, open(f'svm_{ker}_{main_label}.pkl', 'wb'))
                best_acc = acc

def test_best_svm(main_label,ker_list,test_data,test_label):
    """
    测试最好的模型
    Args:
        main_label:
        test_data: 测试数据
        test_label: 测试标签

    Returns:

    """
    class_name = test_label[0]
    test_label = np.where(test_label == main_label, 1, 0)
    print("test_label:", test_label)
    for ker in ker_list:
        print(f"svm use kernel:{ker}")
        svm_model_path = f"./svm_{ker}_{main_label}.pkl"
        svm_model = pickle.load(open(svm_model_path, 'rb'))
        acc = svm_model.score(test_data, test_label)
        print(f"{class_name}-{ker}-预测正确率：{acc * 100}%")





if __name__ == '__main__':
    labels = ["sport","pet","scenery","child"]
    ker_list = ['linear', 'poly', 'rbf', 'sigmoid']
    setup_seed(3407)
    if not os.path.exists('train.txt'):
        write_train_file_to_txt('/home/public/datasets/split_data')

    backbone = resnet50(pretrained=True)
    features = list(backbone.children())[:-2]
    model = torch.nn.Sequential(*features)
    model.eval()


    for label_name in labels:
        train_data,train_label = load_image_data('/home/public/datasets/split_data','train.txt',sample_num=1000,use_cnn=True,model=model,use_pca=True)
        print("-"*6+f"训练类别为{label_name}的模型"+"-"*6)
        train_svm(train_data,train_label,label_name,fold=10)

    test_data,test_label = generate_test_index(use_cnn=True,model=model,use_pca=True)
    test_data_1, test_label_1 = test_data[0:50], test_label[0:50]
    test_data_2,test_label_2 = test_data[51:100],test_label[51:100]
    test_data_3, test_label_3 = test_data[101:150], test_label[101:150]
    test_data_4, test_label_4 = test_data[151:], test_label[151:]
    test_best_svm("scenery",ker_list,test_data_1, test_label_1)
    test_best_svm('child',ker_list,test_data_2,test_label_2)
    test_best_svm("sport",ker_list,test_data_3, test_label_3)
    test_best_svm("pet",ker_list,test_data_4, test_label_4)
