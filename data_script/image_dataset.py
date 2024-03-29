import __main__
from configs.config import *
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
            img_extension = '.jpg'
            text_extension = '.txt'
            self.all_image_paths = glob.glob(path_to_data + f"*/**/*{img_extension}",recursive=True)
            self.all_face_paths = glob.glob(path_to_data + f"*/**/*{text_extension}",recursive=True)
        elif self.dataset_name == "HAND_CRAWL":
            self.all_image_paths = glob.glob(path_to_data + "*/*.jpg")
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
            label = (image_path.split("/")[-3])  
            if label == 'living':
                label = 0
                return label
            if label != 'spoof':
                label = (image_path.split("/")[-4])  # move to spoof
                if label == "spoof":
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
            
            image = cv2.imread(image_path)
            label = self.get_label(image_path)

            image_path_without_extension = os.path.splitext(image_path)[0]
            face_path_without_extension = os.path.splitext(face_path)[0]
            # if image_path_without_extension != face_path_without_extension
            #   find face_path_without_extension == image_path_without_extension
            # else:
            #   skip image
            if image_path_without_extension != face_path_without_extension:
                # Find a matching face_path_without_extension that corresponds to the current image_path_without_extension
                matching_face_path = None
                for index in range(len(self.all_face_paths)):
                    current_face_path = self.all_face_paths[index]
                    current_face_path_without_extension = os.path.splitext(current_face_path)[0]
                    if current_face_path_without_extension == image_path_without_extension:
                        matching_face_path = current_face_path
                        break
            
                if matching_face_path is not None:
                    print(  "    I   "  +  image_path_without_extension)
                    print(  '    F   '    +  str(matching_face_path) + "\n")
                    with open(face_path, 'r') as file:
                        bbox = [next(file).strip() for _ in range(2)]
                    # print("First two lines:", lines_array)
                    # cut the image by the bbox
                    x1, y1 = map(int, bbox[0].split())
                    x2, y2 = map(int, bbox[1].split())      
                    
                    image = cv2.imread(image_path)
                    cropped_image = image[y1:y2, x1:x2] # after cut.



                    # if (cropped_image==0).any():
                    #     pass
                    label = self.get_label(image_path)
                    label = torch.tensor(label)
                    t_image = self.transform(cropped_image)
                    # cv2.imwrite(f'test_image/{index}.jpg' ,cropped_image)
                    # t_label = torch.tensor((label))
                    return t_image, label
                else:
                    pass
        
        elif self.augment == 'val':
            image_path = self.all_image_paths[index]
            face_path = self.all_face_paths[index]
            
            image = cv2.imread(image_path)
            label = self.get_label(image_path)

            image_path_without_extension = os.path.splitext(image_path)[0]
            face_path_without_extension = os.path.splitext(face_path)[0]

            # print(  "    I   "  +  image_path_without_extension)
            # print(  '    F   '    +  face_path_without_extension)
            
            t_image = self.transform(image)
            t_label = torch.tensor(label)
            return t_image, t_label

        elif self.augment == 'test':
            image_path = self.all_image_paths[index] # all image paths
            image = cv2.imread(image_path)
            label = self.get_label(image_path)
            return image, label
        # if self.model_backbone == 'mnv3':





