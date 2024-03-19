from libs import *
from scrfd import SCRFD
from image_dataset import ImageDataset
from liveness_detection import LivenessDetection

class FaceDetector():
    def __init__(self) -> None: 
        # self.data = self.load_data()
        self.path_to_fd_model = "./model/scrfd.onnx"
        self.path_to_fas_model = "./model/fas.onnx"
        self.path_to_one_image = "./data/fake.png"
        self.path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
        self.path_to_data = './data/all_images/'
        self.path_to_labeled_data = './data/images/'

        self.path_to_write = './data/videos/fake_video_frames/'
        self.path_to_video_dir = './data/videos/fake_videos/'
        self.model_format = "onnx"
        self.model = self.load_model()
        # self.dataset = self.load_dataset()
        pass
    def load_model(self):
        scrfd = SCRFD(model_file=self.path_to_fd_model)
        print("loaded: " + str(self.path_to_fd_model))
        scrfd.prepare(-1)
        return scrfd
    
    def load_dataset(self):
        dataset = ImageDataset(self.path_to_labeled_data,
                           image_size= 640,
                           model_format=self.model_format)
        return dataset
    
    def face_detect_one_image(self):
        # path to sample photo: nparray (1,3,x,y)
        img = cv2.imread(self.path_to_one_image)
        #print( "Original shape: " +  str(img.shape))
        if img is not None: # check path
            fd = self.model
            bboxes, kpss = fd.detect(img, 0.5)
            #print("Bounding box list of array: " + str(bboxes))
            # print("Complete one image.")
            return img, bboxes # , kpss
        else:
            #next img
            # print("no image.")
            return 0
        
    def face_detect_image_dir(self,img): 
            if img is not None: # check path
                fd = self.model # format error
                bboxes, kpss = fd.detect(img, 0.5, input_size = (640,640)) # error cv2 
                #print("Bounding box list of array: " + str(bboxes))
                return img, bboxes # , kpss
            else:
                #next img
                print("no image.")
                return 0
        
    def crop_one_face(self,):
        # load img path
        img, bboxes = self.face_detect_one_image()
        cropped_img_list = []
        for bbox in bboxes:            
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            cropped_img = img[y_min:y_max, x_min:x_max]
            #print("Cropped shape: " +  str(cropped_img.shape))
            # multiple faces in one image.
            cropped_img_list.append(cropped_img)
        return cropped_img_list 
    
    def crop_one_face_dir(self,img_dir):
        # open img path
        img, bboxes = self.face_detect_image_dir(img_dir)
        cropped_img_list = []
        for bbox in bboxes:   
            # print(str(bbox))         
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            cropped_img = img[y_min:y_max, x_min:x_max]
            #print("Cropped shape: " +  str(cropped_img.shape))
            # multiple faces in one image.
            cropped_img_list.append(cropped_img)
        return cropped_img_list 
    
    
    def crop_multiple_faces():
        # for face in faces
        
        
        pass
    # format into 128 np array
    def format_cropped_faces(self):
        faces = self.crop_one_face()
        formatted_faces = []
        for face in faces:
            face_image = Image.fromarray(face) # PIL support
            resized_face = face_image.resize((128, 128))
            resized_face = np.expand_dims(resized_face,0)
            face_array = np.array(resized_face).astype(np.float32)
            face_array = np.transpose(face_array, (0,3,1,2))   
            #print("Shape before FAS: "  + str(face_array.shape))     
            formatted_faces.append(face_array)
        return formatted_faces  # list of 128x128 arrays
    
    def format_cropped_faces_dir(self,img_dir):
        faces = self.crop_one_face_dir(img_dir)
        formatted_faces = []
        for face in faces:
            face_image = Image.fromarray(face) # PIL support
            resized_face = face_image.resize((128, 128))
            resized_face = np.expand_dims(resized_face,0)
            face_array = np.array(resized_face).astype(np.float32)
            face_array = np.transpose(face_array, (0,3,1,2))   
            #print("Shape before    FAS: "  + str(face_array.shape))     
            formatted_faces.append(face_array)
        # list of 128x128 arrays in one pic
        return formatted_faces
    
    
    def run_on_img_dir(self, img_dir,):
        formatted_faces = self.format_cropped_faces_dir(img_dir)
        # array of arrays, pass label
        return formatted_faces#, label
    
    def face_detect_folder():
        pass
    def face_detect_one_video():
        pass
    def face_detect_video_folder():
        pass

if __name__ == '__main__':
    # test paths    
    
    fd = FaceDetector()
    # fd.face_detect_one_image()
    # img = ""
    # convert to numpy array(img)
    
    #img_dir = fd.path_to_one_image
    #fd.run_on_img_dir(img_ir)