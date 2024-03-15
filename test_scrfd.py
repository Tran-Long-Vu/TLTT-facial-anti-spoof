from libs import *
from main.scrfd import SCRFD
from image_dataset import ImageDataset

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
        print(img.shape)
        if img is not None: # check path
            fd = self.load_model()
            bboxes, kpss = fd.detect(img, 0.5, input_size = (640, 640))
            print("Bounding box list of array: " + str(bboxes))
            return img, bboxes # , kpss
        else:
            #next img
            return 0
        # folder dir? 
    def crop_faces_check(self,):
        # load img path
        img, bboxes = self.face_detect_one_image()
        cropped_img_list = []
        for bbox in bboxes:            
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_img_list.append(cropped_img)
        # in img, cut the img along the bboxs
        for index, img in enumerate(cropped_img_list):
            file_name = f"cropped_image{index}.jpg"
            print("wrote: " + file_name)
            cv2.imwrite(file_name, cropped_img)
        # save
        
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
    face_detector.crop_faces_check() # outputs a bounding box.
    # cut the image using the bounding box
    # output =  face_detector.run(data)
    # output.reformat()
    
    # FAS inference:
    # fas = LivenessDetection()
    # fas.run(output)
    # fas.metrics()
    