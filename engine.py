from libs import *
from FAR_FRR import FAR_FRR
# train engine

class TestEngine():
    def __init__(self) -> None: 
        pass 
    
    # pth inference with metrics
    def test_pth_fn(self,
                test_loader, 
                model, 
                device, # default to put numpy on cuda.
    ):
        print("\nStart Testing ...\n" + " = "*16)
        model = model.to(device)

        with torch.no_grad():
            model.eval()
            running_loss, running_corrects,  = 0.0, 0.0, 
            running_labels, running_predictions,  = [], [], 
            for images, labels in tqdm.tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                # print("loader in device")
                
                logits = model(images.float())
                loss = F.cross_entropy(logits, labels)

                running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), 
                running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
                running_labels, running_predictions,  = running_labels + labels.data.cpu().numpy().tolist(), running_predictions + torch.max(logits, 1)[1].detach().cpu().numpy().tolist(), 
        
        test_loss, test_accuracy,  = running_loss/len(test_loader.dataset), running_corrects/len(test_loader.dataset), 
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
            "test", 
            test_loss, test_accuracy, 
        ))
        print(metrics.classification_report(
            running_labels, running_predictions, 
            digits = 4, 
        ))

        print("\nFinish Testing ...\n" + " = "*16)
        return {
            "test_loss":test_loss, "test_accuracy":test_accuracy, 
        }
    
    # onnx inference with metrics
    # write the inference code. (with accuracy and loss.)
    def test_onnx_fn(self,
                test_loader, 
                model, 
                device, #default
    ):
        
        print("\nStart Testing ...\n" + " = "*16)
        # check model in device (done)
        onnx_model = model # inference session format
        
        running_loss, running_corrects = 0.0, 0.0
        running_labels, running_predictions = [] ,[]
        test_loss, test_accuracy = 0.0, 0.0
        
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        # loop
        for images, labels, in tqdm.tqdm(test_loader): # batches
            images, labels = images.to(device), labels.to(device)
            input_data = images.float().cpu().numpy() # change to np
            reshaped_input = input_data.reshape(-1,3,128,128) # reshape np
            # Perform inference using the ONNX model. Image by Image
            outputs = onnx_model.run(None, {'actual_input_1': reshaped_input})
            logits = torch.from_numpy(outputs[0]).to(device) # prediction
            loss = F.cross_entropy(logits, labels) #loss
            
            running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
            running_labels, running_predictions,  = running_labels + labels.data.cpu().numpy().tolist(), running_predictions + torch.max(logits, 1)[1].detach().cpu().numpy().tolist(), 
            # count
            if torch.max(logits, 1)[1] == 1 and labels.data == 1:
                true_positive += 1
            elif torch.max(logits, 1)[1] == 0 and labels.data == 0:
                true_negative += 1
            elif torch.max(logits, 1)[1] == 1 and labels.data == 0:
                false_positive += 1
            elif torch.max(logits, 1)[1] == 0 and labels.data == 1:
                false_negative += 1
        
        test_loss, test_accuracy,  = running_loss/len(test_loader.dataset), running_corrects/len(test_loader.dataset), 
        
        # output
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
            "test", 
            test_loss, test_accuracy, 
        ))
        print(metrics.classification_report(
            running_labels, running_predictions, 
            digits = 4, 
        ))
        
        #print("TP: " + str(true_positive))  
        #print("TN: " + str(true_negative))  
        #print("FP: " + str(false_positive)) 
        #print("FN: " + str(false_negative)) 
        
        far =  false_positive / (false_positive + true_negative)  * 100
        frr = false_negative / (false_negative + true_positive) * 100
        
        print("FAR: " + "{:.2f}".format(far) +  "%") 
        print("FRR: " + "{:.2f}".format(frr) +  "%")
        print("HTER: " +  "{:.2f}".format((far + frr)/2) +  "%" )
        
        print("\nFinish Testing ...\n" + " = "*16)
        return {
            "test_loss":test_loss, "test_accuracy":test_accuracy, 
        }