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
    training_dataset = ImageDataset(TRAIN_DATASET,
                                    PATH_TO_TRAIN_DATASET,
                                    MODEL_BACKBONE,
                                    augment = 'val',
                                    )
    # print(PATH_TO_TRAIN_DATASET)

    
    for image, label in training_dataset: # image and label
        # write zero input output size here
        if image.size(2) == 0:
            continue

        # label = torch.tensor(label)
        # print(str(image.shape))
        # print(str(label.shape))
        # # print(str(face))
        # print(str(type(image)))
        # print(str(type(label)))

        # print(   '     label:     ' +  str(label))
        for image_path, face_path in zip(training_dataset.all_image_paths, training_dataset.all_face_paths):
            print(image_path)
            print(face_path)
            print("\n")



    fd = FaceDetector()
    fas = LivenessDetection()