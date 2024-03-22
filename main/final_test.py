from libs import *
from scrfd import SCRFD
from face_detector import *
from image_dataset import ImageDataset
from liveness_detection import LivenessDetection
from video_dataset import VideoDataset
import math
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
class FasSolution():
    def __init__(self) -> None:
        
        self.fas_model_backbone = "rn18" # can select mnv3 or rn18
        
        self.path_to_data = './data/all_images/'
        self.path_to_labeled_data = './data/images/'
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
            self.provider = ["CPUExecutionProvider"]
        elif self.device == torch.device("cuda"):
            self.provider = ["CUDAExecutionProvider"]
        else: 
            print("device issue")

        if self.fas_model_backbone == "mnv3":
            self.fas = LivenessDetection("./model/mnv3-fas.onnx")
            self.fd = FaceDetector()
            self.fd.fas_model_backbone = "mnv3"
            
            
        elif self.fas_model_backbone == "rn18":
            self.fas = LivenessDetection("./model/rn18-fas.onnx")
            self.fd = FaceDetector()
            self.fd.fas_model_backbone = "rn18"
            
            
        else:
            print ("incorrect model")
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

    def run_on_image_dataset(self):
        print("Start Testing ...\n" + " = "*16  + "  with backbone: " +  str(self.fd.fas_model_backbone))
        inference_times = []
        true_labels = []
        predicted_labels = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for image, label in tqdm.tqdm(self.dataset):
            if self.dataset is not None:
                start_time = time.time()
                formatted_faces = self.fd.run_on_img_dir(image)
                if formatted_faces is not None:
                    outputs = self.fas.run_one_img_dir(formatted_faces)
                    
                    if self.fas_model_backbone == 'mnv3':
                        prediction = outputs[0][0]
                        # print("          prediction array:    "   +    str(prediction))
                        if prediction is not None:
                            if label == 0:
                                logit = round(prediction[0])
                            elif label == 1:
                                logit = round(prediction[1])
                            # if prediction[0] < prediction[1]: # one hot  # real
                            #     logit = 0
                            # elif prediction[0] >= prediction[1]: # fake
                            #     logit = 1    
                            
                            if logit == 1 and label == 1:
                                TP += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            if logit == 0 and label == 0:
                                TN += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            if logit == 1 and label == 0:
                                FP += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            if logit == 0 and label == 1:
                                FN += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                        else:
                            continue
                    elif self.fas_model_backbone == 'rn18': # different calculation method
                        prediction = outputs[0] # null error
                        # print("         FAS predicted:    "   + str(prediction))
                        prediction_softmax = np.exp(prediction) / np.sum(np.exp(prediction))
                        print("   softmax prediction array:    "   +    str(prediction_softmax))
                        # if prediction[0] > prediction[1]: # one hot  # fake
                        #     logit = 0
                        # elif prediction[0] <= prediction[1]: # real
                        #      logit = 1
                        if prediction_softmax is not None:
                            if label == 0:
                                logit = round(prediction_softmax[0])
                            elif label == 1:
                                logit = round(prediction_softmax[1])
                                
                            if logit == 1 and label == 1:
                                TP += 1
                                # print("         FAS predicted:    "   + str(logit))
                                # print("         label:    "   + str(label))
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 0 and label == 0:
                                TN += 1
                                # print("         FAS predicted:    "   + str(logit))
                                # print("         label:    "   + str(label))
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 1 and label == 0:
                                FP += 1
                                # print("         FAS predicted:    "   + str(logit))
                                # print("         label:    "   + str(label))
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 0 and label == 1:
                                FN += 1
                                # print("         FAS predicted:    "   + str(logit))
                                # print("         label:    "   + str(label))
                                true_labels.append(label)
                                predicted_labels.append(logit)
                        else:
                            continue
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
        
        print(metrics.classification_report(true_labels, predicted_labels, digits=4))
        
        print("TP: " + "{}".format(TP))
        print("TN: " + "{}".format(TN))
        print("FP: " + "{}".format(FP))
        print("FN: " + "{}".format(FN))
        
        
        far =  FP / (FP + TN)  * 100
        frr = FN / (FN + TP) * 100
        print("FAR: " + "{:.2f}".format(far) +  "%")
        print("FRR: " + "{:.2f}".format(frr) +  "%")
        print("HTER: " +  "{:.2f}".format((far + frr)/2) +  "%" )
        print("\nFinish Testing ...\n" + " = "*16)
        print("average inference time for one image: " +  "{:.2f}".format(average_time)+ " seconds.")
        print("total inference time: " +  "{:.2f}".format(sum(inference_times)) + " seconds.")
    pass

    def run_on_video_dataset(self):
        print("Start Testing ...\n" + " = "*16  + "  with backbone: " +  str(self.fas_model_backbone))
        inference_times = []
        video_TP = 0.01
        video_TN = 0.01
        video_FP = 0.01
        video_FN = 0.01
        predicted_labels = []
        true_labels = []

        # todo - other   ortsess  [] [] []   condition for inferencing with resnet18. 
        
        for frame_array, label in tqdm.tqdm(self.video_dataset):
            if self.video_dataset is not None: 
                start_time = time.time()
                img_TP = 0.01
                img_TN = 0.01
                img_FP = 0.01
                img_FN = 0.01
                for frame in frame_array:
                    if frame is not None:
                        formatted_faces = self.fd.run_on_img_dir(frame)
                        if formatted_faces is not None:
                            outputs = self.fas.run_one_img_dir(formatted_faces)
                            if self.fas_model_backbone == "mnv3":
                                prediction = outputs[0][0]
                                if label == 0:
                                    logit = round(prediction[0])
                                elif label == 1:
                                    logit = round(prediction[1])
                                if logit == 1 and label == 1:
                                    img_TP += 1.0
                                elif logit == 0 and label == 0:
                                    img_TN += 1.0
                                elif logit == 1 and label == 0:
                                    img_FP += 1.0
                                elif logit == 0 and label == 1:
                                    img_FN += 1.0
                            # if self.fas_model_backbone == "rn18":
                            #     prediction = outputs[0]
                            #     if label == 0:
                            #         logit = round(prediction[0])
                            #     elif label == 1:
                            #         logit = round(prediction[1])
                            #     if logit == 1 and label == 1:
                            #         img_TP += 1.0
                            #     elif logit == 0 and label == 0:
                            #         img_TN += 1.0
                            #     elif logit == 1 and label == 0:
                            #         img_FP += 1.0
                            #     elif logit == 0 and label == 1:
                            #         img_FN += 1.0
                        else:
                            formatted_faces = []
                            continue    
            label_0_ratio = (img_TN / img_FP)
            label_1_ratio = (img_TP / img_FN)
            if label_1_ratio > 1.00 and label == 1:
                video_prediction = 1
                predicted_labels.append(video_prediction)
                true_labels.append(label)
            elif label_1_ratio <= 1.00 and label == 1:
                video_prediction = 0
                predicted_labels.append(video_prediction)
                true_labels.append(label)
            elif label_0_ratio >= 1.00 and label == 0: 
                video_prediction = 0
                predicted_labels.append(video_prediction)
                true_labels.append(label)
            elif label_0_ratio < 1.00 and label == 0: 
                video_prediction = 1
                predicted_labels.append(video_prediction)
                true_labels.append(label)
            if video_prediction == 1 and label == 1:
                video_TP += 1.0
            elif video_prediction == 0 and label == 0:
                video_TN += 1.0
            elif video_prediction == 1 and label == 0:
                 video_FP += 1.0
            elif video_prediction == 0 and label == 1:
                video_FN += 1.0
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
        average_time = sum(inference_times) /  len(self.dataset)
        far =  video_FP / (video_FP + video_TN)  * 100
        frr = video_FN / (video_FN + video_TP) * 100
        print(metrics.classification_report(true_labels, predicted_labels, digits=4))
        
        print("     FAR: " + "{:.2f}".format(far) +  "%")
        print("     FRR: " + "{:.2f}".format(frr) +  "%")
        print("     HTER: " +  "{:.2f}".format((far + frr)/2) +  "%" )
        print("average inference time for one frame: " +  "{:.2f}".format(average_time)+ " seconds.")
        print("total inference time all video: " +  "{:.2f}".format(sum(inference_times) / 60) + " minutes.")
        print("\nFinish Testing ...\n" + " = "*16)
        pass
if __name__ == '__main__':
    fas_solution = FasSolution()
    fas_solution.run_on_image_dataset()
    # fas_solution.run_on_video_dataset()
    pass