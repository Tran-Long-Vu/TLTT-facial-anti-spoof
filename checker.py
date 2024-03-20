# data format checker
from libs import *
from main.image_dataset import ImageDataset
import onnxruntime as ort
import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader
if __name__ == '__main__':
    print("cuda: " + str(torch.cuda.is_available()))
    # test paths
    path_to_data_dir = 'data/images/'
    model_format = 'onnx' # TODO: Make global vars into config file. (next week)
    dataset = ImageDataset(path_to_data_dir,
                           image_size= 128,
                           model_format=model_format)
    print("dir:" + path_to_data_dir)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1, 
    )
    print("loaded torch.dataloader " +
          " | model format:  " + str(model_format) + 
          " |  Batches: " + str(len(test_loader)))
    
    # init dataloader
    

    
    
    # debug
    image, label = dataset[0] # first in the batch
    # print(image, "label: " + str(label))
    device = "cpu" # predicfne 
    
    
    print("before transofrm:  " +  str(image.shape))
    #numpy array 4d format. reduce dimentison. 
    image = np.squeeze(image)
    
    # move to tensor then move to device then turn back to np
    transform = tfs.ToTensor()
    # print(transform.)
    # change format back:
    
    # tensor
    tensor_img = transform(image) 
    print(tensor_img.shape)
    tensor_img = tensor_img.to(device) #
    
    
    #print("moved  img to cpu")
    label = 0
    label = torch.tensor(label)
    label_tensor = label.to(device)
    print("turned label to tensor")
    
    # chaneg to np array: 
    np_img = np.array(tensor_img)
    np_img = np.expand_dims(image,0) # create dim
    print("after  np  :  "  + str (np_img.shape))
    
    
    # change label to int
    label = label_tensor.item()
    print("label: " + str(label))
    #print(np_img.shape)

