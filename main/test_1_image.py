# Code to test single image
import numpy as np
import onnxruntime as ort
import onnx
import os
import PIL.Image as Image
import torchvision.transforms as tf
import cv2
# FAS
class LivenessDetection():
    def __init__(self) -> None: 
        pass
    
    def configurations(self): # key info
        pass
    
    @classmethod # load model
    def load_model(self,
                   path_to_model):
        model = onnx.load(path_to_model)
        onnx.checker.check_model(model)
        return model
    
    @classmethod # single image preprocess.
    def pre_processing(self,
                       path_to_image): 
        image = Image.open(path_to_image)
        image = image.resize((128, 128))
        image = np.expand_dims(image,0)
        image_array = np.array(image).astype(np.float32)
        image_array = np.transpose(image_array, (0,3,1,2))        
        return image_array
    
    def single_video_pre_processing(path_to_video):
        cap = cv2.VideoCapture(path_to_video)
        success, image = cap.read()
        count = 0
        while success:
            cv2.imwrite("frame" % count, image)
            success, image = cap.read()
            count+=1
        # images, to list of NP array
        # return NP array
        return 0
            
    @classmethod # video preprocess
    def post_processing(self):
        # input tensor
        # run check format
        pass
    
    @classmethod # run on single img
    def run_on_image(self,
                     path_to_model, 
                     path_to_image): # run
        #model = self.load_model(path_to_model)
        image = self.pre_processing(path_to_image)
        ort_sess = ort.InferenceSession(path_to_model)
        outputs = ort_sess.run(None, {'actual_input_1': image})
        print("Prediction output: " + str(outputs))# print image name / result
        return 0
    
    @classmethod # run on folder
    def run_on_folder(self, 
                      path_to_data,
                      path_to_model):
        images = os.listdir(path_to_data)
        for image in images:
            path_to_image = os.path.join(path_to_data + image)
            self.run_on_image(path_to_model, path_to_image)
        return 0 
    
    @classmethod # run on video
    def run_on_video(self, 
                     path_to_single_video, 
                     path_to_model):
        # read inference
        # 
        # cv2 read frames
        # model inference (frame into dir)
        # 
        return 0
    
    def run_on_one_video(self,
                         path_to_video_data,
                         path_to_model):
        # read list of images
        # for image in list
        # inference.
        # return putput
        return 0

class FaceDetection():
    def __init__(self) -> None:
        pass
    #...

class FASSolutions():
    def __init__(self) -> None:
        pass
    #...
    
if __name__ == '__main__':
    # processor = ... CPU
    path_to_model = "./model/anti-spoof-mn3.onnx"
    path_to_image = './data/real.jpg'
    path_to_video = ''
    path_to_data = './data/images/all/'
    
    obj_test = LivenessDetection() # init class
    #obj_test.load_model(path_to_model)        # run def
    #obj_test.pre_processing(path_to_image)
    #obj_test.run_on_image(path_to_model, path_to_image)
    obj_test.run_on_folder(path_to_data, path_to_model)
    #obj_test.run_on_videos().



