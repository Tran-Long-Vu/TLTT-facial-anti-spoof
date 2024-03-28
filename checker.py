from configs.config import *
from libs import *

from data_script.video_dataset import VideoDataset
from data_script.image_dataset import ImageDataset

from engines.face_detector import FaceDetector
from engines.liveness_detection import LivenessDetection

from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort
import torchvision.transforms as tfs

import sklearn.metrics as metrics
import pandas as pd




if __name__ == '__main__':
    training_dataset = ImageDataset(TEST_DATASET,
                                    PATH_TO_TEST_DATASET,
                                    MODEL_BACKBONE,
                                    augment = 'test',
                                    )
    image, label = training_dataset[1] # image and label
    print(len(training_dataset))
    print(str(image.shape))
    #print(str(face))
    print(str(label))
    print(str(type(image)))
    print(str(type(label)))

    fd = FaceDetector()
    fas = LivenessDetection()




