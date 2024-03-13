# import
from libs import *

# Class init
class ImageDataset(torch.utils.data.Dataset):
    # Init
    def __init__(self,
                 path_to_data_dir,
                 image_size = 128 # default
    ) -> None: 
        self.all_image_paths = glob.glob(path_to_data_dir + "*/*")
        # print(str(self.all_image_paths))
        print("got all image paths")
        self.transform = A.Compose(
                [
                    A.Resize(height = image_size, width = image_size, ), 
                    A.Normalize(mean = (0.491372549, 0.482352941, 0.446666667, ), std = (0.247058824, 0.243529412, 0.261568627, ), ), AT.ToTensorV2(), 
                ]
            )
    
    # Length
    def __len__(self, 
    ):
        return len(self.image_files)

    # Get label
    def get_label(self, image_path):
        label = int(os.path.basename(os.path.dirname(image_path)))
        return label
    
    # get image tensor
    def __getitem__(self, index ):
        image_path = self.all_image_paths[index]
        image = Image.open(image_path)
        image = np.array(image) # np array
        image = self.transform(image = image) # needs numpy array input
        # print(image.size(1))
        label = self.get_label(image_path)
        return image, label