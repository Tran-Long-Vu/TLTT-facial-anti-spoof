# import
from libs import *

# Class init
class ImageDataset(torch.utils.data.Dataset):
    # Init
    def __init__(self,
                 path_to_data_dir,
                 image_size = 128,
                 model_format =  "" # default
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
    # preprocess onnx data
    def transform_onnx(self,
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
        image = Image.open(image_path)
        image = np.array(image) # np array
        if self.model_format == "pth":
           image = self.transform_torch(image = image) # torch tensor format
        if self.model_format == "onnx":
            image = self.transform_onnx(image_path) # numpy array format
        # print(image.size(1))
        label = self.get_label(image_path)
        return image, label