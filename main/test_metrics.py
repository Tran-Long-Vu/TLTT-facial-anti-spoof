# Code to test single image
import numpy as np
import onnxruntime as ort
import onnx
import os
import PIL.Image as Image
import torchvision.transforms as tf
import cv2
import time
# FAS
class LivenessDetection():
    def __init__(self) -> None: 
        pass
    
    def configurations(self): # key info
        pass

    @classmethod # single image preprocess.
    def pre_processing(self):
        return 0
    
    # run on single img
    def run_on_image(self):
        return 0
    
    # run on video. auto delete
    def run_on_video_dir(self):
        return 0
    
    # Inference on 1 video
    def run_on_one_video(self):
        return 0
    
if __name__ == '__main__':
    # test paths
    path_to_model = "./model/anti-spoof-mn3.onnx"
    path_to_image = './data/real.jpg'
    path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
    path_to_data = './data/images/all/'
    path_to_write = './data/videos/fake_video_frames/'
    path_to_video_dir = './data/videos/fake_videos/'
    
    # Tests:
    obj_test = LivenessDetection() # init class
    # obj_test.run_on_image(path_to_model, path_to_image)
    # obj_test.run_on_folder(path_to_data, path_to_model)
    # obj_test.single_video_pre_processing(path_to_video, path_to_write)
    # obj_test.run_on_one_video(path_to_single_video, path_to_write, path_to_model)
    # obj_test.run_on_all_videos(path_to_video_dir, path_to_write, path_to_model )