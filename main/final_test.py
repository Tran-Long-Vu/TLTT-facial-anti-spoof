from libs import *
from scrfd import SCRFD
from face_detector import *
from image_dataset import ImageDataset
from liveness_detection import LivenessDetection
from video_dataset import VideoDataset
class FasSolution():
    def __init__(self) -> None: 
        self.fas = LivenessDetection()
        self.fd = FaceDetector()
        self.path_to_fas_model = "./model/fas.onnx"
        
        self.path_to_image = './data/real.png'
        # self.label = "0"
        self.path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
        self.path_to_data = './data/all_images/'
        self.path_to_labeled_data = './data/images/'
        # self.path_to_write = './data/videos/fake_video_frames/'
        
        self.path_to_video_dir = './data/videos/'
        self.model_format = "onnx"
        
        self.dataset = self.load_dataset()
        
        self.video_dataset = self.load_video_dataset()
        
        # self.dataloader = self.load_dataloader()
        self.provider = ""
        self.cuda = torch.cuda.is_available()
        
        if self.cuda == True:
            print("cuda ready")
            self.device = torch.device("cuda") 
        else:
            print("cpu ready")
            self.device = torch.device("cpu") 

        if self.device == torch.device("cpu"):
            self.provider = ["CPUExecutionProvider"]# handling onnx
        elif self.device == torch.device("cuda"):
            self.provider = ["CUDAExecutionProvider"]
        else: 
            print("device issue")
        
        # a
        # b
        pass
    def load_dataset(self):
        dataset = ImageDataset(self.path_to_labeled_data,
                           model_format=self.model_format)
        return dataset
    
    def load_video_dataset(self):
        video_dataset = VideoDataset(self.path_to_video_dir)
        return video_dataset
    
    def load_dataloader(self):
        dataset = self.load_dataset()
        print("dir:" + self.path_to_labeled_data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 1, 
        )
        print("loaded torch.dataloader " +
              " | model format:  " + str(self.model_format) +
              " |  Batches: " + str(len(loader)))
        return loader
    
    # def run_fas_one_image(self):
    #     fas = self.fas
    #     formatted_faces = self.fd.format_cropped_faces()
    #     for face in formatted_faces:
    #         outputs = fas.run_on_formatted_image(face)
    #         print("Prediction output: " + str(outputs))
    #     return outputs 



    def run_on_image_dataset(self):
        running_loss, running_corrects = 0.0, 0.0
        running_labels, running_predictions = [] ,[]
        test_loss, test_accuracy = 0.0, 0.0
        
        inference_times = []
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for image, label in tqdm.tqdm(self.dataset): # change to dataloader causes error.
            if self.dataset is not None:
                start_time = time.time()
                formatted_faces = self.fd.run_on_img_dir(image)
                
                if formatted_faces is not None:
                    
                    outputs = self.fas.run_one_img_dir(formatted_faces) # format: [0.02825426, 0.9717458 ]], dtype=float32)]
                    prediction = outputs[0][0]
                    if label == 0: 
                        logit = round(prediction[0])
                    elif label == 1:
                        logit = round(prediction[1])
                    
                    if logit == 1 and label == 1:
                        TP += 1
                    if logit == 0 and label == 0:
                        TN += 1
                    if logit == 1 and label == 0:
                        FP += 1
                    if logit == 0 and label == 1:
                        FN += 1
                    
                    
                    
                else:
                    formatted_faces = []
                    continue
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)
            else:
                print("no image loaded")
                return 0
        average_time = sum(inference_times) / len(self.dataset)
        accuracy = (TP + TN) / (TP + TN + FP + FN) 
        far =  FP / (FP + TN)  * 100
        frr = FN / (FN + TP) * 100
        
        
        print("FAR: " + "{:.2f}".format(far) +  "%")
        print("FRR: " + "{:.2f}".format(frr) +  "%")
        print("HTER: " +  "{:.2f}".format((far + frr)/2) +  "%" )
        print("  Accuracy:  " + "{:.2f}".format(accuracy) +  "%")
        print("\nFinish Testing ...\n" + " = "*16)
        print("average inference time for one image: " +  "{:.2f}".format(average_time)+ " seconds.")
        print("total inference time: " +  "{:.2f}".format(sum(inference_times)) + " seconds.")
    
    pass

        
    def run_fas_one_video():
        pass
    def run_on_video_dataset(self):
        inference_times = []
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for frame_array, label in tqdm.tqdm(self.video_dataset):
            if self.video_dataset is not None:
                start_time = time.time()
                for frame in frame_array:
                    if frame is not None: 
                        formatted_faces = self.fd.run_on_img_dir(frame)
                        if formatted_faces is not None:
                            outputs = self.fas.run_one_img_dir(formatted_faces)
                            prediction = outputs[0][0]
                            if label == 0: 
                                logit = round(prediction[0])
                            elif label == 1:
                                logit = round(prediction[1])
                            if logit == 1 and label == 1:
                                TP += 1
                            elif logit == 0 and label == 0:
                                TN += 1
                            elif logit == 1 and label == 0:
                                FP += 1
                            elif logit == 0 and label == 1:
                                FN += 1
                        else:
                            formatted_faces = []
                            continue
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)
            else:
                print("no video set loaded")
                return 0        
        average_time = sum(inference_times) / len(self.dataset)
        accuracy = (TP + TN) / (TP + TN + FP + FN) 
        far =  FP / (FP + TN)  * 100
        frr = FN / (FN + TP) * 100
        
        
        print("FAR: " + "{:.2f}".format(far) +  "%")
        print("FRR: " + "{:.2f}".format(frr) +  "%")
        print("HTER: " +  "{:.2f}".format((far + frr)/2) +  "%" )
        print("  Accuracy:  " + "{:.2f}".format(accuracy) +  "%")
        print("\nFinish Testing ...\n" + " = "*16)
        print("average inference time for one video: " +  "{:.2f}".format(average_time)+ " seconds.")
        print("total inference time all video: " +  "{:.2f}".format(sum(inference_times) / 60) + " minutes.")
        pass
        

if __name__ == '__main__':
    fas_solution = FasSolution()
    # array output
    # fas_solution.run_on_image_dataset() 
    # fas_solution.run_on_one_video()
    fas_solution.run_on_video_dataset()
    pass