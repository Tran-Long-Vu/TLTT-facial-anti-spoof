# test with fd model
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
    def preprocess():
        pass
    def postprocess():
        pass
    def load_dataloader(self,path_to_data_dir,model_format):
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
    # Paths
    path_to_data_dir = "data/images/"
    path_to_fd_model = "./model/scrfd.onnx"
    path_to_fas_model = "./model/fas.onnx"
    model_format = 'onnx'
    provider = ''
    device = torch.device("cpu") 
    if device == torch.device("cpu"):
        provider = ["CPUExecutionProvider"]# handling onnx
    elif device == torch.device("cuda"):
        provider = ["CUDAExecutionProvider"]
    else: 
        print("device issue")
    
    # Tests: Re run raw.
    obj_test = LivenessDetection() 
    test_dataset = obj_test.load_dataloader(path_to_data_dir, model_format) 
    model = obj_test.load_model(path_to_fd_model,
                                model_format,
                                provider ) 
    obj_test.run_test(test_dataset,model, model_format = model_format, device = device)