# import
from libs import *
from PIL import Image# Class init
class ImageDataset(torch.utils.data.Dataset):
    # Init
    def __init__(self,
                 path_to_data_dir,
                 model_format =  "fd" # default
    ) -> None: 
        self.all_image_paths = glob.glob(path_to_data_dir + "*/*")
        self.model_format = model_format
        self.transform = tf.ToTensor()
    # fd transform: 1,3,640,640
    def transform_t(self,
                       img):
        t_image = self.transform(img)
        
        return t_image

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
        # image = Image.open(image_path) 
        # cv2
        image = cv2.imread(image_path)
        label = self.get_label(image_path)
        
        # np array
        # image = np.array(image) 
        # image = np.resize(image, (640,640,3))
        
        # TODO - image to tensor
        t_image = self.transform_t(image)
        t_label = torch.tensor(label)
        
        # print(image.size(1))
        
        return image, label