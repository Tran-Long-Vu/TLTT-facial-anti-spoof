from libs import *
from engines.scrfd import SCRFD
from data_script.image_cvpr23_dataset import ImageDataset
from config import *
import sklearn.metrics as metrics
import pandas as pd

# todo - configs file

class FasTrainer():
    def __init__(self) -> None:
        self.fas_model_backbone = "rn18" # change between 'rn18' and 'mnv3'
        self.path_to_data = './data/crawl_test/images/'
        self.path_to_video_dir = './data/crawl_test/videos/'
        self.path_to_save_frames = "./data/crawl_test/frames/"
        self.model_format = "onnx"
    
    # init image dataset format
    def load_image_dataset(self):
        dataset = ImageDataset(self.path_to_data,
                           model_format=self.model_format)
        return dataset
    
    # init video dataset format
    def load_video_dataset(self):
        return 0
    
    # Dataloader
    def load_image_dataloader(self):
        
        return 0
        # Dataloader
    def load_video_dataloader(self):
        return 0
    # run printing attack dataset
    def train_on_image_dataset(self):
        
        pass
    def train_on_video_dataset(self):
        
        pass
    
    def visualize():
        pass
    # run on replay attack
    def run_on_video_dataset(self):
        pass
    
# Uncomment to test other attacks.
if __name__ == '__main__':
    fas_trainer = FasTrainer()
    # fas_trainer.train_printing_atack
    
    pass