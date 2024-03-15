from libs import *
import os
from PIL import Image
import numpy as np
from scrfd import *

def delete_all_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_all_in_directory(file_path)
            os.rmdir(file_path)

class FacialDetector():
    def __init__(self) -> None: 
        pass
    
    def configurations(self): # key info
        pass
    
    @classmethod # load model
    def load_model(self,
                   path_to_model,
                   model_format = ""):
        
        if model_format =='onnx':
            import onnxruntime
            onnx_model = onnxruntime.InferenceSession(path_to_model)
            print( 'Loaded:' + str(onnx_model._model_path))
            return onnx_model
        # torch
        if model_format == 'pth':
            import torch
            import torchvision.models as models
            # model = models.resnet()
            pth_model = torch.load(path_to_model, map_location=torch.device('cpu'))
            # 
            pth_model.eval()
            print( 'Loaded: pth')
            return 0
        else:
            print("model error")
            return 0
        # TODO: if format == "jit"
    
    @classmethod # single image preprocess.
    def pre_processing(self):
        # iamge
        pass               
    
    # process a single video, 30fps default
    def single_video_pre_processing(self):
        pass
    
    @classmethod # run on single img
    def run_on_image(self): # run        
        return 0
    
    @classmethod # run on folder
    def run_on_folder():
        return 0 
    
    # Inference on 1 video
    def run_on_one_video():
        return 0

    # TODO: run all videos
    def run_on_all_videos():
        return 0

class FaceDetection():
    def __init__(self) -> None:
        pass
    #...

class FASSolutions():
    def __init__(self) -> None:
        pass
    #...

# Run
if __name__ == '__main__':
    # test paths
    path_to_model = "./model/face-detection-0100.pth" # dictionary error. Predefine the model first. 
    path_to_fd_model = "./model/scrfd.onnx"
    path_to_image = './data/real.jpg'
    path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
    path_to_data = './data/all_images/'
    path_to_write = './data/videos/fake_video_frames/'
    path_to_video_dir = './data/videos/fake_videos/'
    model_format = "onnx"

    # Tests:
    obj_test = FacialDetector() # init class
    obj_test.load_model(path_to_fd_model, model_format)
    # obj_test.run_on_image(path_to_model, path_to_image, model_format)
    # obj_test.run_on_folder(path_to_data, path_to_model, model_format)
    # obj_test.single_video_pre_processing(path_to_video, path_to_write, model_format)
    # obj_test.run_on_one_video(path_to_single_video, path_to_write, path_to_model, model_format)
    # obj_test.run_on_all_videos(path_to_video_dir, path_to_write, path_to_model, model_format)