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
from resnet import combined_model, resnet50
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Image IS Score")
    parser.add_argument(
        "--path_gt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--path_gen", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--path_src", type=str, required=True, help="path to the model checkpoint"
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
            path = self.imgs[index]
            if "_age_1" in path:
                path = path[:-10]+".png"
                # print(path)
            return self.transform(Image.open(self.imgs[index])), path

        def __len__(self):
            return len(self.imgs)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    model = combined_model(2)
    model = resnet50()
    model.eval()
    model.cuda()
    dataset1 = IgnoreLabelDataset(args.path_gt)
    dataset2 = IgnoreLabelDataset(args.path_src)
    dataset3 = IgnoreLabelDataset(args.path_gen)
    print ("Calculating Sim Score...")
    correct = 0
    target_dic = {}
    target_path = {}
    with torch.no_grad():
        for i, data in enumerate(dataset1):
            data, path = data
            name = os.path.basename(path)[:3]
            if name not in target_dic:
                target_dic[name] = []
                target_path[name] = []
            out=  model(data.cuda()[None,...]).squeeze()
            target_dic[name].append(out.detach())
            target_path[name].append(path)
        for key in target_dic:
            target_dic[key] = torch.stack(target_dic[key], dim=0)
            
        src_target = {}
        src_dict = {}
        for i, data in enumerate(dataset2):
            data, path = data
            name = os.path.basename(path)[:3]
            out=  model(data.cuda()[None,...]).squeeze().unsqueeze(0)
            target = target_dic[name]
            
            out_n = nn.functional.normalize(out, dim=1)
            target_n = nn.functional.normalize(target, dim=1)
            
            sim = out_n@ target_n.T 
            index = torch.argmax(sim, dim=1)
            file = os.path.basename(path)
            dif = target[index] - out
            if file not in src_dict:
                src_dict[file] = dif 
                src_target[file] = out
            print(file, target_path[name][index])
                
                
        sum_all = 0
                
        for i, data in enumerate(dataset3):
            data, path = data
            file = os.path.basename(path)
            out=  model(data.cuda()[None,...]).squeeze().unsqueeze(0)
            # print(out.shape)
            src = src_target[file]
            ori_dif = src_dict[os.path.basename(path)].squeeze().unsqueeze(0)
            cur_dif = out - src
            ori_dif = nn.functional.normalize(ori_dif, dim=1)
            cur_dif = nn.functional.normalize(cur_dif, dim=1)
            sim = (cur_dif@ori_dif.T).squeeze()
            sum_all += sim
            
        print(sum_all/len(dataset3))
    # print(correct/len(dataset))
        
