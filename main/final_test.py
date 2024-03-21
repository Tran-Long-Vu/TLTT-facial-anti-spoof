from libs import *
from scrfd import SCRFD
from face_detector import *
from image_dataset import ImageDataset
from liveness_detection import LivenessDetection
from video_dataset import VideoDataset
import math
import sklearn.metrics as metrics
class FasSolution():
    def __init__(self) -> None: 
        self.fas = LivenessDetection()
        self.fd = FaceDetector()
        self.path_to_fas_model = "./model/fas.onnx"
        
        # self.path_to_image = './data/real.png'
        # self.label = "0"
        # self.path_to_single_video = './data/videos/fake_videos/20240312_021946.mp4'
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
        print("Start Testing ...\n" + " = "*16)
        inference_times = []
        
        video_TP = 0.01
        video_TN = 0.01
        video_FP = 0.01
        video_FN = 0.01
        
        count = 0
        threshold = 0.70
        
        for frame_array, label in tqdm.tqdm(self.video_dataset):
            if self.video_dataset is not None: # video dir loop
                # init 
                # on new video, reset counter
                start_time = time.time()
                real_frames = 0.01
                fake_frames = 0.01
                
                single_video_accuracy = 0.01
                
                img_TP = 0.01
                img_TN = 0.01
                img_FP = 0.01
                img_FN = 0.01
                
                for frame in frame_array: # frame in single video
                    # total_frames = len(frame_array)
                    # reset frame counter
                    if frame is not None: 
                        
                        formatted_faces = self.fd.run_on_img_dir(frame)
                        if formatted_faces is not None:
                            outputs = self.fas.run_one_img_dir(formatted_faces)
                            prediction = outputs[0][0]
                            
                            if label == 0: 
                                logit = round(prediction[0])
                            elif label == 1:
                                logit = round(prediction[1])
                            # debugging binary prediction
                            if logit == 1 and label == 1:
                                img_TP += 1.0
                            elif logit == 0 and label == 0:
                                img_TN += 1.0
                            elif logit == 1 and label == 0:
                                img_FP += 1.0
                            elif logit == 0 and label == 1:
                                img_FN += 1.0

                        else:
                            formatted_faces = []
                            continue
                        # record single video prediction
            
            # add far, frr
            
            # count += 1            
            label_0_ratio = (img_TN / img_FP)
            label_1_ratio = (img_TP / img_FN)
            # append into list
            
            #print("single video TP  " + "{:.2f}".format(img_TP))
            #print("single video TN  " +  "{:.2f}".format(img_TN))
            #print("single video FP  " +  "{:.2f}".format(img_FP)) 
            #print("single video FN  " +  "{:.2f}".format(img_FN))
            #
            # print("     single_video_accuracy:  " + "{:.2f}".format((single_video_accuracy)) + "  on  label  "  + str(label) )
            
            
            # one-hot returns low accuracy.
            # threshold returns high accuracy.
            if label_1_ratio > 1.00 and label == 1:
                video_prediction = 1
            elif label_1_ratio <= 1.00 and label == 1:
                video_prediction = 0
            elif label_0_ratio >= 1.00 and label == 0: 
                video_prediction = 0
            elif label_0_ratio < 1.00 and label == 0: 
                video_prediction = 1
            
            if video_prediction == 1 and label == 1:
                video_TP += 1.0
            elif video_prediction == 0 and label == 0:
                video_TN += 1.0
            elif video_prediction == 1 and label == 0:
                 video_FP += 1.0
            elif video_prediction == 0 and label == 1:
                video_FN += 1.0
            # video loop
            # print("    video predicted:"  +  str(video_prediction))
            # print("    label"  + str(label))
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            # test the test
            #if count >= 2:
            #    break
        # count more metrics
        # AUC
        # F1
        # TODO - Add array memory to use sklearn.metrics
        # ...
        # ... 
        # ... 
        average_time = sum(inference_times) /  len(self.dataset)
        
        # recount far frr hter
        
        accuracy = (video_TP + video_TN) / (video_TP + video_TN + video_FP + video_FN) 
        far =  video_FP / (video_FP + video_TN)  * 100
        frr = video_FN / (video_FN + video_TP) * 100
        
        #print("video TP  " + "{:.2f}".format(video_TP))
        #print("video TN  " +  "{:.2f}".format(video_TN))
        #print("video FP  " +  "{:.2f}".format(video_FP))
        #print("video FN  " +  "{:.2f}".format(video_FN))
        
        # print(metrics.classification_report(label, video_prediction, digits))
        accuracy = (video_TP + video_TN) / (video_TP + video_TN + video_FP + video_FN)
        recall = video_TP / (video_TP + video_FN)
        precision = video_TP / (video_TP + video_FP)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        print("     accuracy:  " + "{:.2f}".format(accuracy*100) +  "%")
        print("     recall: " + "{:.2f}".format(recall*100) +  "%")
        print("     precision: " + "{:.2f}".format(precision*100) +  "%")
        print("     f1_score: " +  "{:.2f}".format((f1_score*100)) +  "%" )
        
        print("     FAR: " + "{:.2f}".format(far) +  "%")
        print("     FRR: " + "{:.2f}".format(frr) +  "%")
        print("     HTER: " +  "{:.2f}".format((far + frr)/2) +  "%" )
        
        
        print("average inference time for one frame: " +  "{:.2f}".format(average_time)+ " seconds.")
        print("total inference time all video: " +  "{:.2f}".format(sum(inference_times) / 60) + " minutes.")
        
        print("\nFinish Testing ...\n" + " = "*16)
        pass
        

if __name__ == '__main__':
    fas_solution = FasSolution()
    # array output
    # fas_solution.run_on_image_dataset()
    fas_solution.run_on_video_dataset()
    pass