# import
from libs import *

# Class init
class ImageDataset(torch.utils.data.Dataset):
    # Init
    def __init__(self,
                 path_to_data_dir,
                 image_size = 640,
                 model_format =  "fd" # default
    ) -> None: 
        self.all_image_paths = glob.glob(path_to_data_dir + "*/*")
        self.model_format = model_format
        # print(str(self.all_image_paths))
        print("got all image paths")
        self.transform_torch = A.Compose(
                    [
                        A.Resize(height = image_size, width = image_size, ), 
                        A.Normalize(mean = (0.491372549, 0.482352941, 0.446666667, ), std = (0.247058824, 0.243529412, 0.261568627, ), ), AT.ToTensorV2(), 
                    ]
                )    
    # fd transform: 1,3,640,640
    def transform_fd(self,
                       path_to_image): 
        image = cv2.imread(path_to_image)
      
        return image
    
    # fas transform
    def transform_fas(self,
                       path_to_image): 
        image = Image.open(path_to_image)
        image = image.resize((128, 128))
        image = np.expand_dims(image,0)
        image_array = np.array(image).astype(np.float32)
        image_array = np.transpose(image_array, (0,3,1,2))        
        return image_array
    
    # Length
    def __len__(self, 
    ):
        return len(self.all_image_paths)

    # Get label
    def get_label(self, image_path):
        label = int(os.path.basename(os.path.dirname(image_path)))
        return label
    
    # get image tensor
    def __getitem__(self, index ):
        image_path = self.all_image_paths[index]
        # PIL
        image = Image.open(image_path) 
        # np array
        image = np.array(image) 
        if self.model_format == "fas":
           image = self.transform_fas(image_path) # torch tensor format
        if self.model_format == "fd":
            image = self.transform_fd(image_path) # numpy array format
        # print(image.size(1))
        label = self.get_label(image_path)
        return image, label