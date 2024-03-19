from libs import *
import os
from PIL import Image
import numpy as np

class LivenessDetection():
    def __init__(self) -> None: 
        self.path_to_fas_model = "./model/fas.onnx"
        self.path_to_image = './data/real.png'
        self.path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
        self.path_to_data = './data/all_images/'
        self.path_to_write = './data/videos/fake_video_frames/'
        self.path_to_video_dir = './data/videos/fake_videos/'
        self.model_format = "onnx"
        self.model = self.load_model()
    
    def configurations(self): # key info
        pass

    def load_model(self):
        if self.model_format == 'onnx':
            import onnxruntime
            onnx_model = onnxruntime.InferenceSession(self.path_to_fas_model)
            print( 'Loaded:' + str(onnx_model._model_path))
            return onnx_model
        if self.model_format == 'pth':
            import torch
            pth_model = torch.load(self.path_to_fas_model)
            pth_model.eval()
            print( 'Loaded: pth')
            return pth_model
        else:
            print("model error")
            return 0
        # TODO: if format == "jit"
    
    def pre_processing(self,): 
        image = Image.open(self.path_to_image)
        image = image.resize((128, 128))
        image = np.expand_dims(image,0)
        image_array = np.array(image).astype(np.float32)
        image_array = np.transpose(image_array, (0,3,1,2))        
        return image_array
    
    # process a single video, 30fps default
    def single_video_pre_processing(self):
        pass
    
    # def run_on_one_image(self): # run
    #    #model = self.load_model(path_to_fas_model)
    #    image = self.pre_processing()
    #    ort_sess =  self.model 
    #    outputs = ort_sess.run(None, {'actual_input_1': image})
    #    print("Prediction output: " + str(outputs))# print image name / result
    #    return 0
    
    def run_on_formatted_image(self, image):
        ort_sess = self.model
        outputs = ort_sess.run(None, {'actual_input_1': image})
        #print("Prediction output: " + str(outputs))# print image name / result
        return 0
    
    def run_one_img_dir(self, face):
        ort_sess = self.model
        outputs = ort_sess.run(None, {'actual_input_1': face})
        #print("Prediction output: " + str(outputs))
        return outputs
    
    def run_on_folder(self,):
        images = os.listdir(self.path_to_data)
        for image in images:
            path_to_image = os.path.join(self.path_to_data + image)
            self.run_on_image(self.path_to_fas_model, path_to_image, model_format=self.model_format)
        return 0 
    
    # Inference on 1 video
    def run_on_one_video(self,
    ):
        return 0



# Run
if __name__ == '__main__':
    # test paths
    
    # Tests:
    obj_test = LivenessDetection() # init class
    # obj_test.load_model(path_to_model, model_format)
    # obj_test.run_on_one_image()
    # obj_test.run_on_folder(path_to_data, path_to_model, model_format)
    # obj_test.single_video_pre_processing(path_to_video, path_to_write, model_format)
    # obj_test.run_on_one_video(path_to_single_video, path_to_write, path_to_model, model_format)
    # obj_test.run_on_all_videos(path_to_video_dir, path_to_write, path_to_model, model_format)