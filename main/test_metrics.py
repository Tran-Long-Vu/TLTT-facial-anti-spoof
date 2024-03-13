import os, sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from engine import TestEngine
from libs import *
from image_dataset import ImageDataset
# FAS

class LivenessDetection():
    def __init__(self) -> None: 
        pass
    
    def configurations(self): # key config saves
        pass
    
    # return preprocessing albumentations input(tensor)
    def preprocess():
        pass
    
    # post process onnx output into metrics
    def postprocess():
        pass
    
    #load torch utils dataset
    def load_dataloader(self,data_dir): # loader
        # init DataLoader
        test_loader = torch.utils.data.DataLoader(
            ImageDataset(
                path_to_data_dir = data_dir, 
                image_size=128
            ), 
            batch_size = 32, 
        )
        return test_loader
    
    # load model
    @classmethod # load model
    def load_model(self,
                   path_to_model,
                   format = ""):
        
        if format =='onnx':
            import onnxruntime
            onnx_model = onnxruntime.InferenceSession(path_to_model)
            print( 'Loaded:' + str(onnx_model._model_path))
            return onnx_model
        if format == 'pth':
            import torch
            pth_model = torch.load(path_to_model)
            pth_model.eval()
            print( 'Loaded: pth')
            return pth_model
        # TODO: if format == "jit"
    
    # run inference with onnx model
    @classmethod
    def run_test(self,
                 test_loader,
                 model,
                 format = ""
                 ): # run test
        # init egnine()
        engine = TestEngine()
        if format == "onnx":
            engine.test_onnx_fn(test_loader=test_loader,
                       model = model,
                       device=torch.device("cpu")
        )
        if format == "pth":
            engine.test_pth_fn(test_loader=test_loader,
                       model = model,
                       device=torch.device("cpu")
        )
        return 0
if __name__ == '__main__':
    # test paths
    data_dir = "/data/images/"
    path_to_onnx_model = "./model/anti-spoof-mn3.onnx"
    path_to_pth_model = ""
    model_format = 'onnx'
    #path_to_image = './data/real.jpg'
    #path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
    #path_to_data = './data/images/all/'
    #path_to_write = './data/videos/fake_video_frames/'
    #path_to_video_dir = './data/videos/fake_videos/'
    
    
    # Tests:
    obj_test = LivenessDetection() # init class
    test_loader = obj_test.load_dataloader(data_dir)
    model = obj_test.load_model(path_to_onnx_model, format = model_format)
    obj_test.run_test(test_loader,model, format = model_format)