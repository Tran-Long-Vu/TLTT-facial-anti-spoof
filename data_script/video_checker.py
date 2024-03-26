# data format checker
from libs import *
from data_script.video_dataset import VideoDataset
import onnxruntime as ort
import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader
from engines.face_detector import FaceDetector
from engines.liveness_detection import LivenessDetection


if __name__ == '__main__':
    print("cuda: " + str(torch.cuda.is_available()))
    # test paths
    path_to_single_video = 'data/videos/'
    print("checker video    |    loading a single video: ")
    
    fd = FaceDetector()
    fas = LivenessDetection()
    # for every frame in frames:
        # run face detector(frame)
        # process bboxes. record frame bboxes
        # run fas. record
    
    
        # return width height of the bounding box
        # return confidence score of facial spoofing ()
        # reture
        
        # test output:
        # frame --- width --- height --- confidence score that its a spoof.
        # 1 ---- 444 ---- 555 ----- 0.9(fake one).
        # 2 ---- 444 ---- 555 ----- 0.9(fake one).
        # average: 435 --- 535 ---- 0.8