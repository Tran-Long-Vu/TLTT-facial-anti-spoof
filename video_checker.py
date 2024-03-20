# data format checker
from libs import *
from main.video_dataset import VideoDataset
import onnxruntime as ort
import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    print("cuda: " + str(torch.cuda.is_available()))
    # test paths
    
    path_to_video_dir = 'data/videos/'
    
    model_format = 'onnx' 
    
    dataset = VideoDataset(path_to_video_dir)
    
    print("dir:" + path_to_video_dir)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1, 
    )
    
    print("loaded torch.dataloader " +
          " | model format:  " + str(model_format) + 
          " |  Batches: " + str(len(test_loader)))
    
    print("checker video    |    loading a single video: ")
    frame_array = dataset[3]
    print(frame_array)
    # print(frame_array.shape)
    # print(type(frame_array))
    # print(frame_array)
