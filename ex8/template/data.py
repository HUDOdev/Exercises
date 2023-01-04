from torch.utils.data import Dataset


class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
        
    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        raise NotImplementedError
