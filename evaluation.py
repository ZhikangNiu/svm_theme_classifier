# -*- coding: utf-8 -*-
# @Time    : 2022-09-22 22:28
# @Author  : Zhikang Niu
# @FileName: evaluation.py
# @Software: PyCharm

"""
1. 读取视频和他的类别，初始化一个预测类
2. 读取预测的结果
3. 计算准确率
"""
import os
from train import write_file_to_txt
from main import ThemePredict
def eval(theme_predict,video_path):
    # theme_predict = ThemePredict(video_path=video_path,extract_frame_mode=extract_frame_mode,time_gap=time_gap,use_cnn_ectract=use_cnn_extract)
    pred = theme_predict.predict(video_path)
    pred = sorted(pred.items(),reverse=True,key=lambda item:item[1])
    return pred[0][0]

if __name__ == '__main__':
    if not os.path.exists("./video_list.txt"):
        write_file_to_txt("/home/public/nas_datasets/荣耀交付数据集2022年9月15日/",'./video_list.txt',mode='.mp4',label_split_index=-1)
    theme_predict = ThemePredict()
    with open("./video_list.txt","r") as f:
        lines = f.readlines()
    f.close()

    with open("result.txt", "w") as res:
        for line in lines:
            video_path = line.split(" ")[0]
            label = line.split(" ")[1].rstrip()
            pred = eval(theme_predict,video_path)
            res.write(f"{video_path} {label} {pred}\n")


