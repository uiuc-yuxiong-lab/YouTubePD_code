import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import os 
import torchvision.transforms as transforms
from PIL import Image
import argparse
from resnet import combined_model
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Image IS Score")
    parser.add_argument(
        "--path", type=str, required=True, help="path to the model checkpoint"
    )
    args = parser.parse_args()
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, root):
            self.root = root 
            self.imgs = os.listdir(root)
            self.imgs = [os.path.join(root, img) for img in self.imgs]
            self.transform =  transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
            
        def __getitem__(self, index):
            return self.transform(Image.open(self.imgs[index]))

        def __len__(self):
            return len(self.imgs)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    model = combined_model(2)
    model.load_state_dict(torch.load('pd_binary.pth'))
    model.eval()
    model.cuda()
    dataset = IgnoreLabelDataset(args.path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)
    print ("Calculating CLS Score...")
    correct = 0
    for i, data in enumerate(dataloader):
        out=  model(data.cuda())
        pred = torch.argmax(out, dim=1)
        # print(pred)
        correct += torch.sum(pred == 1)
    print(correct/len(dataset))
        
