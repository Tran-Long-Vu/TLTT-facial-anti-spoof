from libs import *
from scrfd import SCRFD
from image_dataset import ImageDataset
from liveness_detection import LivenessDetection

class FaceDetector():
    def __init__(self) -> None:
        # self.data = self.load_data()
        self.path_to_fd_model = "./model/scrfd.onnx"
        self.path_to_labeled_data = './data/images/'
        
        self.fas_model_backbone = ""
        
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
            if image is not None: 
                fd = self.model
                bboxes, kpss = fd.detect(image,0.3)
                return bboxes 
            else:
                print("no image.")
                return 0

    def crop_one_face_dir(self,image, bboxes):
        cropped_faces = []
        for bbox in bboxes:
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            face = image[y_min:y_max, x_min:x_max]
            cropped_faces.append(face)
        return cropped_faces


    def format_cropped_images_dir(self, cropped_faces):
        # TODO: array of faces found in one image.
        if len(cropped_faces) != 0:
            face = cropped_faces[0]  # first face
            face = np.expand_dims(face,0)
            if self.fas_model_backbone == "mnv3":
                face = np.resize(face, (1,3,128,128))
            elif self.fas_model_backbone == "rn18":
                face = np.resize(face, (1,3,256,256))
            face = np.array(face).astype(np.float32)
            return face
        else:
            # print( "                   FD found no face. " )
            pass
    
    def run_and_record_frame_video(self, frame):
        bboxes = self.face_detect_image_dir(frame)
        if type(bboxes) is  int:
            pass
        else:
            bbox = bboxes[0]
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            
            width = x_max - x_min
            height = y_max - y_min
            
            face = frame[y_min:y_max, x_min:x_max]
            face = np.expand_dims(face,0)
            if self.fas_model_backbone == "mnv3":
                face = np.resize(face, (1,3,128,128))
            elif self.fas_model_backbone == "rn18":
                face = np.resize(face, (1,3,256,256))
            face = np.array(face).astype(np.float32)
            return face, width, height
            
    def run_on_img_dir(self, image,):
        
        if image is not None:
            bboxes = self.face_detect_image_dir(image)
            if len(bboxes) is not None:
                face = self.crop_one_face_dir(image, bboxes)
                formatted_face = self.format_cropped_images_dir(face)
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
    fd = FaceDetector()