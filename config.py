BATCH_SIZE = 16
LOSS = ""
LEARNING_RATE = 0.001
NO_EPOCHS = 5
DROPOUT = False
# OPTIMIZER = "adam"
MODEL_BACKBONE= "rn18"
MODEL_NAME = 'rn18fas'
ATTACK_TYPE = 'printing'

INFERENCE_DEVICE = 'CUDA'
TRAINING_DEVICE = 'CUDA'

TRAIN_DATASET = 'CVPR23'
TEST_DATASET = 'HAND_CRAWL'

# PATH_TO_IMAGES_DATASET = './data/dataset/crawl_test/images/'
PATH_TO_VIDEOS_DATASET = './data/dataset/crawl_test/videos/'
PATH_TO_SINGLE_VIDEO = ''
PATH_TO_TRAIN_DATASET = 'data/datasets/CVPR23/train/'
PATH_TO_TEST_DATASET = './data/dataset/crawl_test/images/'
PATH_TO_FAS_MODEL = ''
PATH_TO_FD_MODEL =''
PATH_TO_STATE_DICT = 'model/rn18-fas-ckp.pth'
PATH_TO_ONNX_FAS = 'model/rn18-fas.onnx'
PATH_TO_SAVE_CHECKPOINT = 'checkpoints'
