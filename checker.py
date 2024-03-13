# data format checker
from libs import *
from image_dataset import ImageDataset
import onnxruntime as ort

if __name__ == '__main__':
    # test paths
    path_to_data_dir = 'data/images/'
    dataset = ImageDataset(path_to_data_dir,
                           image_size= 128)
    # debug
    image, label = dataset[0]
    print(image, "label: " + str(label))
