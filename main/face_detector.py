from libs import *
from scrfd import SCRFD
from image_dataset import ImageDataset
from liveness_detection import LivenessDetection

class FaceDetector():
    def __init__(self) -> None: 
        # self.data = self.load_data()
        self.path_to_fd_model = "./model/scrfd.onnx"
        self.path_to_fas_model = "./model/fas.onnx"
        self.path_to_one_image = "./data/real.png"
        self.path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
        self.path_to_data = './data/all_images/'
        self.path_to_write = './data/videos/fake_video_frames/'
        self.path_to_video_dir = './data/videos/fake_videos/'
        self.model_format = "onnx"
        self.model = self.load_model()
        pass
    def load_model(self):
        scrfd = SCRFD(model_file=self.path_to_fd_model)
        print("loaded: " + str(self.path_to_fd_model))
        scrfd.prepare(-1)
        return scrfd
    def load_data():
        dataset = ImageDataset()
        return dataset
    def face_detect_one_image(self):
        # path to sample photo: cv2.
        img = cv2.imread(self.path_to_one_image)
        print( "Original shape: " +  str(img.shape))
        if img is not None: # check path
            fd = self.model
            bboxes, kpss = fd.detect(img, 0.5, input_size = (640, 640))
            print("Bounding box list of array: " + str(bboxes))
            return img, bboxes # , kpss
        else:
            #next img
            return 0
        # folder dir? 
    def crop_faces_test_1(self,):
        # load img path
        img, bboxes = self.face_detect_one_image()
        cropped_img_list = []
        for bbox in bboxes:            
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            cropped_img = img[y_min:y_max, x_min:x_max]
            print("Cropped shape: " +  str(cropped_img.shape))
            cropped_img_list.append(cropped_img)
        return cropped_img_list
    def format_cropped_faces(self):
        faces = self.crop_faces_test_1()
        formatted_faces = []
        for face in faces:
            face_image = Image.fromarray(face) # PIL support
            resized_face = face_image.resize((128, 128))
            resized_face = np.expand_dims(resized_face,0)
            face_array = np.array(resized_face).astype(np.float32)
            face_array = np.transpose(face_array, (0,3,1,2))   
            print("Shape before FAS: "  + str(face_array.shape))     
            formatted_faces.append(face_array)
        return formatted_faces  # list of 128x128 arrays
    
    def run_raw_fas_one_image(self):
        # call fas
        fas = LivenessDetection()
        formatted_faces = self.format_cropped_faces()
        for face in formatted_faces:
            outputs = fas.run_on_formatted_image(face)
            print("Prediction output: " + str(outputs))
        pass 
    
    def face_detect_folder():
        pass
    def face_detect_one_video():
        pass
    def face_detect_video_folder():
        pass

if __name__ == '__main__':
    # test paths    
    # Data preprocess
    # load data()
    # data = ImageDataset.preprocess()
    
    # Face Detector
    face_detector = FaceDetector()
    face_detector.run_raw_fas_one_image() # outputs a bounding box.
    # TODO: FACE DETECTOR TAKES IN LABEL AND RETURNS LABEL AFTER INFERENCE
    # cut the image using the bounding box
    # output =  face_detector.run(data)
    # output.reformat()
    # fas = LivenessDetection()
    #fas.run_on_one_image()
    # FAS inference:
    # fas = LivenessDetection()
    # fas.run(output)
    # fas.metrics()
    