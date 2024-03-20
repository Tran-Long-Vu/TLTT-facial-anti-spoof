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
        self.model_name = "scrfd"
        self.model_format = "onnx"
        self.model = self.load_model()
        # self.dataset = self.load_dataset()
        pass
    def load_model(self):
        if self.model_name == "scrfd":
            scrfd = SCRFD(model_file=self.path_to_fd_model)
            print("loaded: " + str(self.path_to_fd_model))
            scrfd.prepare(-1)
            return scrfd
        else:
            return 0
    
    def load_dataset(self):
        dataset = ImageDataset(self.path_to_labeled_data,
                           image_size= 640,
                           model_format=self.model_format)
        return dataset

    def face_detect_image_dir(self,image): 
            if image is not None: # check path
                fd = self.model
                # auto resize?
                bboxes, kpss = fd.detect(image,0.5) # error cv2 
                #print("Bounding box list of array: " + str(bboxes))
                return bboxes # , kpss
            else:
                #next img
                print("no image.")
                return 0

    def crop_one_face_dir(self,image, bboxes):
        # open img path
        # img, bboxes = self.face_detect_image_dir(image)
        
        cropped_faces = []
        for bbox in bboxes:
            # print(str(bbox))
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            face = image[y_min:y_max, x_min:x_max]
            #print("Cropped shape: " +  str(cropped_img.shape))
            # multiple faces in one image.
            cropped_faces.append(face)
        return cropped_faces

    def crop_multiple_faces():
        # for face in faces
        pass

    def format_cropped_faces_dir(self, cropped_faces):
        # cropped_img_list = self.crop_one_face_dir(image)
        # TODO: array of faces found in one image.
        # formatted_faces = []
        if len(cropped_faces) != 0:
            face = cropped_faces[0]  # first face
                # error: cv2 resize
                # format for fas
            
            face = np.expand_dims(face,0)
            face = np.resize(face, (1,3,128,128))
            face = np.array(face).astype(np.float32)
            # print ("face" + str(face.shape))
            # face = np.transpose(face, (0,3,1,2))   
            # print("Sha    FAS: "  + str(face.shape))     
            # formatted_faces.append(face)
            return face
        else:
            # print( "FD found no face. " )
            pass

    def run_on_img_dir(self, image,):
        
        if image is not None:
            
            bboxes = self.face_detect_image_dir(image)
            if len(bboxes) is not None:
                face = self.crop_one_face_dir(image, bboxes)
                formatted_face = self.format_cropped_faces_dir(face)
                return formatted_face
            else:
                print("cannot bound any face. poor image format")
                return []
        else:
            print("no image")
            return []
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