from libs import *
from scrfd import SCRFD







if __name__ == '__main__':
    # test paths
    path_to_fd_model = "./model/scrfd.onnx"
    path_to_fas_model = "./model/fas.onnx"
    path_to_image = './data/real.jpg'
    path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
    path_to_data = './data/all_images/'
    path_to_write = './data/videos/fake_video_frames/'
    path_to_video_dir = './data/videos/fake_videos/'
    model_format = "onnx"
        
    # Data preprocess
    data = Dataset.preprocess()
    
    # Face Detector
    # face_detector = FaceDetection()
    # output =  face_detector.run(data)
    # output.reformat()
    
    # FAS inference:
    # fas = LivenessDetection()
    # fas.run(output)
    # fas.metrics()
    