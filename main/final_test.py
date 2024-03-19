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
        self.label = "0"
        
        self.path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
        self.path_to_data = './data/all_images/'
        self.path_to_labeled_data = './data/images/'
        # self.path_to_write = './data/videos/fake_video_frames/'
        self.path_to_video_dir = './data/videos/fake_videos/'
        self.model_format = "onnx"
        self.dataset = self.load_dataset()
        self.dataloader = ''
        self.provider = ""
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
                           image_size= 640,
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
    
    def print_benchmark_output(self):
        pass



    def run_on_image_dataset(self):
        for image, label in self.dataset:
            # image format is nparray.
            formatted_faces, label = self.fd.run_on_img_dir(image, label)      
            for face in formatted_faces:
                output = self.fas.run_one_img_dir(face)
                print("prediction: "+  str(output) + "  label: " + str(label)) # benchmark.
            #   face.append(predictions) 
        #   fas.benchmark(output)
        # return benchmark. 
        pass
    

        
    def run_fas_one_video():
        pass
    
    
    def run_fas_video_dataset():
        pass

if __name__ == '__main__':
    fas_solution = FasSolution()
    # array output
    fas_solution.run_on_image_dataset() #
    pass