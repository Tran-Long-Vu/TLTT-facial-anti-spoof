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
    # hard code test
    def load_dataloader(self,path_to_data_dir,model_format): # loader
        # init Dataset
        dataset = ImageDataset(path_to_data_dir,
                           image_size= 128,
                           model_format=model_format)
        print("dir:" + path_to_data_dir)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 1, 
        )
        print("loaded torch.dataloader " +
              " | model format:  " + str(model_format) +
              " |  Batches: " + str(len(test_loader)))
        return test_loader
    
    # load model
    @classmethod # load model
    def load_model(self,
                   path_to_model,
                   model_format = "",
                   provider = ["CPUExecutionProvider"] # default
                   ):
        
        if model_format =='onnx':
            import onnxruntime
            onnx_model = onnxruntime.InferenceSession(path_to_model, 
                                                      providers = provider)
            print( 'Loaded:' + str(onnx_model._model_path) + " into " + str(ort.get_device()))
            return onnx_model
        if model_format == 'pth':
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
                 model_format,
                 device
                 ): # run test
        # init egnine()
        engine = TestEngine()
        if model_format == "onnx":
            engine.test_onnx_fn(test_loader=test_loader,
                       model = model,
                       device=device
        )
        if model_format == "pth":
            engine.test_pth_fn(test_loader=test_loader,
                       model = model,
                       device=device
        )
        return 0
if __name__ == '__main__':
    # test paths
    path_to_data_dir = "data/images/"
    path_to_onnx_model = "./model/anti-spoof-mn3.onnx"
    path_to_pth_model = ""
    model_format = 'onnx'
    provider = ''
    device = torch.device("cpu") 
    if device == torch.device("cpu"):
        provider = ["CPUExecutionProvider"]# handling onnx
    elif device == torch.device("cuda"):
        provider = ["CUDAExecutionProvider"]
    else: 
        print("device issue")
        
    #path_to_image = './data/real.jpg'
    #path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
    #path_to_data = './data/images/all/'
    #path_to_write = './data/videos/fake_video_frames/'
    #path_to_video_dir = './data/videos/fake_videos/'
    
    
    # Tests:
    obj_test = LivenessDetection() # init class
    test_dataset = obj_test.load_dataloader(path_to_data_dir, model_format) # call dataset
    model = obj_test.load_model(path_to_onnx_model,
                                model_format,
                                provider ) # call self.load_model
    obj_test.run_test(test_dataset,model, model_format = model_format, device = device) # call engine