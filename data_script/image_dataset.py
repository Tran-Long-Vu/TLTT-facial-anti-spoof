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
        
        self.dataset_name = dataset_name
        self.model_backbone = model_backbone
        self.augment = augment
        if self.dataset_name == "CVPR23":
            self.all_image_paths = glob.glob(path_to_data + "*/**/*.jpg",recursive=True)
            self.all_face_paths = glob.glob(path_to_data + "*/**/*.txt",recursive=True)
        elif self.dataset_name == "HAND_CRAWL":
            self.all_image_paths = glob.glob(path_to_data + "**/*.jpg")
        if augment == 'train':
            self.transform = tf.Compose([
                            tf.ToTensor(),
                            tf.Resize((256, 256)),
                            tf.RandomHorizontalFlip(),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            tf.ColorJitter(brightness=0.2, contrast=0.2),
                            tf.RandomRotation(30),
                        ])
        elif augment == 'val':
            self.transform = tf.Compose([
                                tf.ToTensor(),
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                tf.Resize((256, 256)),
                            ])
    
    #length
    def __len__(self, 
    ):
        return len(self.all_image_paths)

    # Get label
    def get_label(self, image_path):
        if self.dataset_name == "CVPR23":
            label = (image_path.split("\\")[1])  
            if label == 'living':
                label = 0
            if label == 'spoof':
                label = 1
            return label
        elif self.dataset_name == "HAND_CRAWL":
            label = int(os.path.basename(os.path.dirname(image_path)))
            return label
    
    # get image tensor
    def __getitem__(self, index ):
        if self.augment == 'train':
            image_path = self.all_image_paths[index]
            face_path = self.all_face_paths[index]
            #todo - read bounding boxes before formatting. 
            # take the 1st two lines in txt. [1,2,3,4]
            with open(face_path, 'r') as file:
                bbox = [next(file).strip() for _ in range(2)]
            # print("First two lines:", lines_array)
            # cut the image by the bbox
            x1, y1 = map(int, bbox[0].split())
            x2, y2 = map(int, bbox[1].split())           
            
            image = cv2.imread(image_path)
            cropped_image = image[y1:y2, x1:x2]
            label = self.get_label(image_path)
            t_image = self.transform(cropped_image)
            t_label = torch.tensor(label)
            return t_image, t_label
        
        
        elif self.augment == 'test':
            image_path = self.all_image_paths[index]
            image = cv2.imread(image_path)
            label = self.get_label(image_path)
            return image, label
        # if self.model_backbone == 'mnv3':





