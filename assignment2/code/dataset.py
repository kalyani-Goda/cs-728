import torch

class STSDataset(torch.utils.data.Dataset):
    """
    Dataset class to help in creating a batch of data.
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        text1 = self.data[index]['sentence1']
        text2 = self.data[index]['sentence2']
        label = self.data[index]['label']
        return (text1, text2), label

    def __len__(self):
        return len(self.data)
