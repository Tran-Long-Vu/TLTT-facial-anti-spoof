# Code to test single image
import numpy as np
import onnxruntime as ort
import onnx
import os
import PIL.Image as Image
import torchvision.transforms as tf
import cv2
import time
# FAS

def delete_all_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_all_in_directory(file_path)
            os.rmdir(file_path)

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
    
    # process a single video, 30fps default
    def single_video_pre_processing(self, path_to_video, path_to_write):
        if not os.path.exists(path_to_write):
            os.makedirs(path_to_write)
        try:
            os.mkdir(path_to_write)
        except OSError:
            pass
        # Log the time
        time_start = time.time()
        # Start capturing the feed
        cap = cv2.VideoCapture(path_to_video)
        # Find the number of frames
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print ("Number of frames: ", video_length)
        count = 0
        print ("Converting video..\n")
        # Start converting the video
        while cap.isOpened():
            # Extract the frame
            ret, frame = cap.read()
            if not ret:
                continue
            # Write the results back to output location.
            cv2.imwrite(path_to_write + "/%#05d.jpg" % (count+1), frame)
            count = count + 1
            # If there are no more frames left
            if (count > (video_length-1)):
                # Log the time again
                time_end = time.time()
                # Release the feed
                cap.release()
                # Print stats
                print ("Done extracting frames.\n%d frames extracted" % count)
                print ("It took %d seconds for conversion." % (time_end-time_start))
                break
    
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
    
    @classmethod # run on video. auto delete
    def run_on_video_dir(self):
        # read inference
        # cv2 read frames
        # model inference (frame into dir)
        return 0
    
    # Inference on 1 video
    def run_on_one_video(self, 
                         path_to_video,
                         path_to_write,
                         path_to_model
    ):
        # preprocess
        self.single_video_pre_processing(path_to_video,
                                         path_to_write
        )
        # folder
        self.run_on_folder(path_to_write,
                           path_to_model
        )
        # delete all
        delete_all_in_directory(path_to_write)
        return 0

    # TODO: run all videos
    def run_on_all_videos(self,
                                path_to_video_dir,
                                path_to_write,
                                path_to_model,
    ):
        # loop through the folder
        videos = os.listdir(path_to_video_dir)
        for video in videos:
            path_to_single_video = os.path.join(path_to_video_dir + video)
            print(path_to_single_video)
            self.run_on_one_video(
                              path_to_single_video,
                              path_to_write,
                              path_to_model
            )
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
    path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
    path_to_data = './data/images/all/'
    path_to_write = './data/videos/fake_video_frames/'
    path_to_video_dir = './data/videos/fake_videos/'
    # init obj
    obj_test = LivenessDetection() # init class
    # obj_test.run_on_image(path_to_model, path_to_image)
    # obj_test.run_on_folder(path_to_data, path_to_model)
    # obj_test.single_video_pre_processing(path_to_video, path_to_write)
    #obj_test.run_on_one_video(path_to_single_video, path_to_write, path_to_model)
    obj_test.run_on_all_videos(path_to_video_dir, path_to_write, path_to_model )