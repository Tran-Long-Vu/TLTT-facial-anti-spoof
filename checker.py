# data format checker
from data_script.config import *
from libs import *
from data_script.video_dataset import VideoDataset
from data_script.image_dataset import ImageDataset

import onnxruntime as ort
import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader
from engines.face_detector import FaceDetector
from engines.liveness_detection import LivenessDetection

if __name__ == '__main__':
    training_dataset = ImageDataset(TRAIN_DATASET,
                                    PATH_TO_TRAIN_DATASET,
                                    MODEL_BACKBONE,
                                    augment = 'train',
                                    )
    image, label = training_dataset[2] # image and label
    print(str(image.shape))
    #print(str(face))
    print(str(label))
    print(str(type(image)))
    print(str(type(label)))

