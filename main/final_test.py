from libs import *
from scrfd import SCRFD
from face_detector import *
from image_dataset import ImageDataset
from liveness_detection import LivenessDetection

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
        self.path_to_video_dir = './data/videos/fake_videos/'
        self.model_format = "onnx"
        self.dataset = self.load_dataset()
        self.dataloader = ""
        self.provider = ""
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
    
    def run_fas_one_image(self):
        fas = self.fas
        formatted_faces = self.fd.format_cropped_faces()
        for face in formatted_faces:
            outputs = fas.run_on_formatted_image(face)
            print("Prediction output: " + str(outputs))
        return outputs 
    
    def benchmark():
        
        pass



    def run_on_image_dataset(self):
        running_loss, running_corrects = 0.0, 0.0
        running_labels, running_predictions = [] ,[]
        test_loss, test_accuracy = 0.0, 0.0
        
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        batch_logits = []
        batch_labels = []
        
        
        for image, label in tqdm.tqdm(self.dataset): # change to dataloader causes error.
            image, label = image.to(self.device), label.to(self.device) 
            
            
            # reformat
            image = np.array(image)
            image = np.resize(image, (640,640,3))
            label = label.item()
            # print("    image    " + str(image.shape))
            # print("    label    " + str(label))
            if image is not None: 
                formatted_faces = self.fd.run_on_img_dir(image)
                for face in formatted_faces: # run evaluation
                    outputs = self.fas.run_one_img_dir(face)
                    logits = torch.from_numpy(outputs[0]).to(self.device) # prediction
                    
                    batch_logits.append(logits)
                    batch_labels.append(label)                    
                    loss = F.cross_entropy(logits, label) #loss
        # Check if any logits and labels were accumulated
        if len(batch_logits) > 0 and len(batch_labels) > 0:
            # Concatenate logits and labels within the batch
            batch_logits = torch.cat(batch_logits)
            batch_labels = torch.tensor(batch_labels)

            # Calculate loss for the accumulated logits and labels
            loss = F.cross_entropy(batch_logits, batch_labels)

        running_loss, running_corrects,  = running_loss + loss.item()*image.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == label.data).item(), 
        running_labels, running_predictions,  = running_labels + image.data.cpu().numpy().tolist(), running_predictions + torch.max(logits, 1)[1].detach().cpu().numpy().tolist(), 
        

        test_loss, test_accuracy,  = running_loss/len(self.dataset), running_corrects/len(self.dataset), 
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
        "test", 
        test_loss, test_accuracy, 
    ))
        
        print("\nFinish Testing ...\n" + " = "*16)
        return {
            "test_loss":test_loss, "test_accuracy":test_accuracy, 
        }



        
    def run_fas_one_video():
        pass
    
    
    def run_fas_video_dataset():
        pass

if __name__ == '__main__':
    fas_solution = FasSolution()
    # array output
    fas_solution.run_on_image_dataset() #
    pass