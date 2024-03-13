from libs import *
# train engine

class TestEngine():
    def __init__(self) -> None: 
        pass 
    
    # test fn
    def test_pth_fn(self,
                test_loader, 
                model, 
                device = torch.device("cpu"), 
    ):
        print("\nStart Testing ...\n" + " = "*16)
        model = model.to(device)

        with torch.no_grad():
            model.eval()
            running_loss, running_corrects,  = 0.0, 0.0, 
            running_labels, running_predictions,  = [], [], 
            for images, labels in tqdm.tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)

                logits = model(images.float())
                loss = F.cross_entropy(logits, labels)

                running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
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
        
    def test_onnx_fn(self,
                test_loader, 
                model, 
                device = torch.device("cpu"), 
    ):
        print("\nStart Testing ...\n" + " = "*16)
        # TODO: Write onnx inference with metrics logging
        
        

        print("\nFinish Testing ...\n" + " = "*16)
        return {
            "test_loss":test_loss, "test_accuracy":test_accuracy, 
        }

