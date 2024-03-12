# Code to test single image
import numpy as np
import onnxruntime as ort
import onnx
import os
import PIL.Image as Image
import torchvision.transforms as tf
# FAS
class LivenessDetection():
    def __init__(self) -> None: # save data
        pass
    def configurations(self): # key info
        pass
    @classmethod
    def load_model(self, path_to_model):
        model = onnx.load(path_to_model)
        onnx.checker.check_model(model)
        #print("loaded model")
        #input_names = [input.name for input in model.graph.input]
        #print("input name" + str(input_names))
        return model
    @classmethod
    def pre_processing(self,path_to_image):
        image = Image.open(path_to_image)
        image = image.resize((128, 128))
        image = np.expand_dims(image,0)
        image_array = np.array(image).astype(np.float32)
        image_array = np.transpose(image_array, (0,3,1,2))
        #image_array.transpose(2,0,1)
        
        #transform = tf.Compose([tf.PILToTensor()])
        #tensor = transform(image)
        
        #print("converted image")
        return image_array
        pass
    def post_processing(self):
        # input tensor
        # run check format
        pass
    @classmethod
    def run_on_image(self,
                     path_to_model, 
                     path_to_image): # run
        model = self.load_model(path_to_model)
        image = self.pre_processing(path_to_image)
        # onnx runtime load model
        ort_sess = ort.InferenceSession(path_to_model)
        outputs = ort_sess.run(None, {'actual_input_1': image})
        print("final output: " + str(outputs))# print result
        return 0
    def run_on_folder(self):
        pass 
    def run_on_video(self):
        pass

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
    path_to_image = './data/10.jpg'
    path_to_video = ''
    obj_test = LivenessDetection() # init class
    #obj_test.load_model(path_to_model)        # run def
    #obj_test.pre_processing(path_to_image)
    obj_test.run_on_image(path_to_model, path_to_image)
