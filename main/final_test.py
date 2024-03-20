from libs import *
from scrfd import SCRFD
from face_detector import *
from image_dataset import ImageDataset
from liveness_detection import LivenessDetection

class FasSolution():
    def __init__(self) -> None: 
        self.fas = LivenessDetection()
        self.fd = FaceDetector()
        self.path_to_fas_model = "./model/fas.onnx"
        
        self.path_to_image = './data/real.png'
        # self.label = "0"
        
        self.path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
        self.path_to_data = './data/all_images/'
        self.path_to_labeled_data = './data/images/'
        # self.path_to_write = './data/videos/fake_video_frames/'
        self.path_to_video_dir = './data/videos/fake_videos/'
        self.model_format = "onnx"
        self.dataset = self.load_dataset()
        self.dataloader = self.load_dataloader()
        self.provider = ""
        self.cuda = torch.cuda.is_available()
        if self.cuda == True:
            print("cuda ready")
            self.device = torch.device("cuda") 
        else:
            print("cpu ready")
            self.device = torch.device("cpu") 

        if self.device == torch.device("cpu"):
            self.provider = ["CPUExecutionProvider"]# handling onnx
        elif self.device == torch.device("cuda"):
            self.provider = ["CUDAExecutionProvider"]
        else: 
            print("device issue")
        
        # a
        # b
        pass
    def load_dataset(self):
        dataset = ImageDataset(self.path_to_labeled_data,
                           model_format=self.model_format)
        return dataset
    
    def load_dataloader(self):
        dataset = self.load_dataset()
        print("dir:" + self.path_to_labeled_data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 1, 
        )
        print("loaded torch.dataloader " +
              " | model format:  " + str(self.model_format) +
              " |  Batches: " + str(len(loader)))
        return loader
    
    def run_fas_one_image(self):
        fas = self.fas
        formatted_faces = self.fd.format_cropped_faces()
        for face in formatted_faces:
            outputs = fas.run_on_formatted_image(face)
            print("Prediction output: " + str(outputs))
        return outputs 
    
    def benchmark():
        
        pass



    def run_on_image_dataset(self):
        running_loss, running_corrects = 0.0, 0.0
        running_labels, running_predictions = [] ,[]
        test_loss, test_accuracy = 0.0, 0.0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        for image, label in tqdm.tqdm(self.dataset): # change to dataloader causes error.
            if len(self.dataset) is not None:
                
                # tensor
                # image, label = image.to(self.device), label.to(self.device)
                
                # reformat
                # cv2 format
                # image = image.transpose((1,2,0))
                # image = cv2.resize(image, (640,640))
                # image = np.transpose(image, (2,0,1))
                # image = np.resize(image, (640,640))
                # label = label.item()
                
                # check
                # print("    image    " + str(image.shape))
                # print("    type    " + str(type(image)))
                # print("    label    " + str(label))
                # input: (1,3,640,640)
                formatted_faces = self.fd.run_on_img_dir(image)
                
                
                # run evaluation
                # print(str(formatted_faces)) # error: no data found
                
                # check null
                if formatted_faces is None:
                    formatted_faces = []
                    continue 
                    # raise ValueError("Invalid NumPy array.")
                else:
                    self.fas.run_one_img_dir(formatted_faces)
                    
                    
            
            else:
                print("no image loaded")
                return 0
            

        
    def run_fas_one_video():
        pass
    def run_fas_video_dataset():
        pass

if __name__ == '__main__':
    fas_solution = FasSolution()
    # array output
    fas_solution.run_on_image_dataset() #
    pass