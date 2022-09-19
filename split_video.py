# -*- coding: utf-8 -*-
# @Time    : 2022-09-18 10:23
# @Author  : Zhikang Niu
# @FileName: split_video.py
# @Software: PyCharm
import ffmpeg
import cv2
import av
import os


"""
1. 提取关键帧 I帧
2. 每隔固定的帧数提取帧
3. 
"""

def extract_video_keyframe(video_path, save_path):
    """
    提取视频的关键帧
    Args:
        video_path: 输入的视频路径：'./video/download1.mp4'
        save_path: 保存关键帧的路径：eg:./save_path/video_name/0001.jpg

    Returns:

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        print(f"视频帧数：{stream.frames}")
        stream.codec_context.skip_frame = 'NONKEY'

        for frame in container.decode(stream):
            print(frame)
            frame.to_image().save(
                os.path.join(save_path, '{:04d}.jpg'.format(frame.pts)),
                quality=80
            )
    container.close()


def extract_time_gap_frame(video_path,save_path,time_gap=10):
    """
    每个特定的帧提取视频的画面帧
    Args:
        video_path: 输入的视频路径：'./video/download1.mp4'
        save_path: 保存关键帧的路径：eg:./save_path/video_name/0001.jpg
        time_gap: 隔多少帧，默认10帧

    Returns:

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with av.open(video_path) as container:
        stream = container.streams.video[0].frames
        frame_count = 0
        print(f"视频共有{stream}帧")
        for frame in container.decode(video=0):
            if frame_count % time_gap == 0:
                print(f"已处理：{frame_count}/{stream},当前帧：{frame.index}")
                frame.to_image().save(
                    os.path.join(save_path, '{:04d}.jpg'.format(frame_count))
                )
            frame_count += 1
    container.close()

if __name__ == '__main__':
    # video_path = './video/child/download2.mp4'
    # video_class = video_path.split('/')[-2]
    # video_name = video_path.split('/')[-1].split('.')[0]
    #
    # save_path = './split_data/'+os.path.join(video_class,video_name)+'/'
    video_dataset_root = '/home/public/nas_datasets/荣耀交付数据集2022年9月15日/'
    for video_class in os.listdir(video_dataset_root):
        video_class_path = os.path.join(video_dataset_root,video_class)
        for video_name in os.listdir(video_class_path):
            if video_name.endswith('.mp4'):
                video_path = os.path.join(video_class_path,video_name)
                save_path = '/home/public/datasets/split_data/'+os.path.join(video_class,video_name)+'/'
                extract_time_gap_frame(video_path,save_path,time_gap=10)
    # extract_time_gap_frame(video_path,save_path,10)
    # extract_video_keyframe(video_path, save_path)