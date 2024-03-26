import __main__
from config import *
# import
from libs import *
import torch
class ImageDataset(torch.utils.data.Dataset):
    # Init
    def __init__(self,
                 dataset_name,
                 path_to_data,
                 model_backbone,
                 augment,
    ) -> None: 
        self.all_image_paths = glob.glob(path_to_data + "*/**/*.jpg",recursive=True)
        self.dataset_name = dataset_name
        self.model_backbone = model_backbone
        self.augment = augment

    #length
    def __len__(self, 
    ):
        return len(self.all_image_paths)

    # Get label
    def get_label(self, image_path):
        label = (image_path.split("\\")[1])  
        if label == 'living':
            label = 0
        if label == 'spoof':
            label = 1
        return label
    
    # get image tensor
    def __getitem__(self, index ):
        image_path = self.all_image_paths[index]
        print(image_path)
        image = cv2.imread(image_path)
        label = self.get_label(image_path)
        # TODO - image to cuda tensor
        # todo - add conditionals for training and testing scripts.
        # t_image = self.transform_t(image)
        # t_label = torch.tensor(label)
        return image, label





