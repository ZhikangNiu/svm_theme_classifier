# -*- coding: utf-8 -*-
# @Time    : 2022-09-22 19:24
# @Author  : Zhikang Niu
# @FileName: main.py
# @Software: PyCharm

"""
1. 获取视频。进行抽帧，得到测试的图片集合
2. 对测试图片进行预测，得到预测结果，为概率和类标的一个字典
3. 对所有预测结果的得分进行排序，得到最终的结果
"""
import os
from split_video import extract_time_gap_frame,extract_video_keyframe
from train import load_image_data
import glob
import pickle
from torchvision.models import resnet50
import torch
class ThemePredict(object):
    def __init__(self,video_path,extract_frame_mode="time_gap",time_gap=10,use_cnn_ectract=True):
        self.mode = extract_frame_mode
        self.video_path = self._get_video_path(video_path)
        self.model_scenery = pickle.load(open(f'svm_linear_scenery.pkl', 'rb'))
        self.model_pet = pickle.load(open(f'svm_linear_pet.pkl', 'rb'))
        self.model_child = pickle.load(open(f'svm_sigmoid_child.pkl', 'rb'))
        self.model_sport = pickle.load(open(f'svm_linear_sport.pkl', 'rb'))


        self.time_gap = time_gap
        self.save_path = f"./extract_frame{self.mode}_{self.video_path.split('/')[-1].split('.')[0]}"
        print(f"抽帧图片将会保存在{self.save_path}文件夹中，程序运行结束将会删除")
        self._extract_frame()
        if use_cnn_ectract:
            backbone = resnet50(pretrained=True)
            features = list(backbone.children())[:-2]
            self.model = torch.nn.Sequential(*features)
            self.model.eval()

    def _get_video_path(self,video_path):
        assert os.path.exists(video_path), "视频路径不存在"
        assert video_path.endswith(".mp4"), "视频格式不正确，请输入mp4格式视频"
        assert self.mode in ["time_gap","key_frame"], "抽帧模式不正确，请输入time_gap或者key_frame"
        return video_path

    def _extract_frame(self):
        if self.mode == "time_gap":
            extract_time_gap_frame(self.video_path,save_path=self.save_path,time_gap=self.time_gap)
        elif self.mode == 'key_frame':
            extract_time_gap_frame(self.video_path,save_path=self.save_path)
        print("抽帧完成")

    def predict(self,rmdir=True):
        image_data = load_image_data(self.save_path,mode="infer",model=self.model,use_cnn=True,use_pca=True)
        length = len(image_data)
        pred_scenery = self.model_scenery.predict(image_data)
        pred_pet = self.model_pet.predict(image_data)
        pred_child = self.model_child.predict(image_data)
        pred_sport = self.model_sport.predict(image_data)
        score_pred = {
            "scenery":pred_scenery.sum()/length,
            "pet":pred_pet.sum()/length,
            "child":pred_child.sum()/length,
            "sport":pred_sport.sum()/length
        }
        if rmdir:
            os.system(f"rm -rf {self.save_path}")

        return score_pred

if __name__ == '__main__':
    video_path = "./test/9.mp4"
    theme_predict = ThemePredict(video_path,extract_frame_mode="time_gap",time_gap=10,use_cnn_ectract=True)
    score = theme_predict.predict()
    score = sorted(score.items(),reverse=True,key=lambda item:item[1])
    print("该视频属于",score[0][0])