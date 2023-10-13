import os
import sys
import warnings
from tqdm import tqdm
import argparse
import math
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchmetrics.classification import F1Score, AUROC, Accuracy

from sklearn.metrics import balanced_accuracy_score
import random
import torchvision.transforms.functional as F
from resnet import resnet50
import pandas
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def contrast(self, kpts, img):
    cri = torch.nn.CrossEntropyLoss().cuda()
    q = self.kpt_net(kpts)  # N, C
    k = self.im_net(img).squeeze()
    q = torch.nn.functional.normalize(q,dim=1)
    k =  torch.nn.functional.normalize(k,dim=1).detach()
    logits = q@k.T
    label = torch.arange(q.shape[0]).cuda().long()
    logits/=0.07 
    # print(logits.shape)
    return cri(logits,label)

eps = sys.float_info.epsilon

def metrics(logits, target, num_class):
    if num_class>2:
        f1 = F1Score(task="multiclass", num_classes=num_class)
        mf1 = F1Score(task="multiclass", average="macro", num_classes=num_class)
        auroc = AUROC(task="multiclass", num_classes=num_class)
    else:
        f1 = F1Score(task="binary")
        mf1 = F1Score(task="multiclass",average="macro",num_classes=2)
        auroc = AUROC(task="binary")
    # acc_m = Accuracy(task="multiclass",num_classes=num_class, average="macro")


    predicts = torch.argmax(logits,dim=1)

    final_f1 = f1(predicts,target)
    final_mf1=  mf1(predicts,target)
    top1 = (predicts==target).sum()/predicts.shape[0]
    # print(logits)
    if num_class==2:
        logits=logits[torch.arange(logits.shape[0]),target]
    # print(torch.arange(logits.shape[0]),logits)
    final_auroc = auroc(logits,target)
    total_mse =((predicts-target)**2).float().mean()

    class_mse = torch.zeros(num_class)
    class_recall = torch.zeros(num_class)
    predicts = predicts.float()
    for i in range(num_class):
        class_mse[i]=((predicts-target)**2)[target==i].mean()
        class_recall[i]=((predicts==target)[target==i]).sum()/(target==i).sum()
        macro_mse = class_mse.mean()
        macro_recall = class_recall.mean()
    return final_f1, final_mf1, top1, final_auroc, total_mse, class_mse, class_recall, macro_mse, macro_recall
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pd_path', type=str, default='../data/PD_DET/')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    # parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')

    return parser.parse_args()

def get_coner(data):
    max_size = 256
    x1 = max(data.min(axis=0)[0],0)
    y1 = min(data.min(axis = 0)[1],max_size)
    x2 = max(data.max(axis=0)[0],0)
    y2 = min(data.max(axis=0)[1],max_size)
    w = x2-x1
    h = y2-y1
    if w*h<40*40:
        x1 = max(0,x1-w/4)
        x2 = min(max_size,x2+w/4)
        y1 = max(0,y1-h/4)
        y2 = min(max_size,y2+h/4)    
    # print([x1,y1,x2,y2])
    return np.array([x1,y1,x2,y2])

class combined_model(nn.Module):
    def __init__(self,num_class, img_net, num_samples = 8):
        super(combined_model, self).__init__()
        self.num_samples = num_samples
        self.img_net = img_net
        self.im_cls = nn.Sequential(
            nn.Linear(2048, num_class)
        )
        self.kpts_cls = nn.Sequential(
            nn.Linear(212, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_class)
        )
        self.audio_cls = nn.Sequential(
            nn.Linear(3840, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_class)
        )
        self.bn_im = nn.BatchNorm1d(num_class)
        self.bn_kpt = nn.BatchNorm1d(num_class)
        self.bn_audio = nn.BatchNorm1d(num_class)
        self.cri = torch.nn.CrossEntropyLoss()
        
    
    def forward(self,data):
        targets = data["label"].cuda()
        
        imgs = data["img"].cuda().reshape(data["img"].shape[0]*self.num_samples,data["img"].shape[1]//self.num_samples,*data["img"].shape[2:])
        kpts = data["kpt"].cuda().reshape(data["kpt"].shape[0]*self.num_samples,-1,*data["kpt"].shape[2:])
        with torch.no_grad():
            imgs = self.img_net(imgs).squeeze()
        imgs = self.im_cls(imgs).squeeze()
        img_out = self.bn_im(imgs)
        kpts_out = self.bn_kpt(self.kpts_cls(kpts).squeeze())
        audio_out = self.bn_audio(self.audio_cls(data["audio"].cuda()).squeeze())
        return {"img":img_out, "kpt":kpts_out, "audio":audio_out},targets
        # return img_out, kpts_out, audio_out, targets
    

    
class PDDataSet(data.Dataset):
    def __init__(self, data_path, phase, im_net, transform = None, num_class = 2, num_samples = 8):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        self.num_samples = num_samples
        with open(os.path.join(data_path,phase+".csv")) as f:
            lines = f.readlines()
            # print(lines)
            self.paths = [os.path.basename(x.split(" ")[0])[:-4] for x in lines]
            self.label = np.array([int(x.split(" ")[1]) for x in lines]) 
        self.adjust_class(self.label,num_class)
        self.img_features = self.get_img_features(im_net)
        self.kpts_features = self.get_kpts_features()
        self.audio_features = self.get_audio_features()
    
    @torch.no_grad()
    def get_img_features(self,im_net):
        img_dir = os.path.join(self.data_path,f'{self.phase}_set',"imgs")
        imgs = os.listdir(img_dir)
        img_features = []
        for i in range(len(self.paths)):
            path = self.paths[i]
            cur_imgs = [x for x in imgs if path in x]
            if len(cur_imgs)<1:
                raise Exception(f"No image found in {path}")
            img_input = []
            for img in cur_imgs:
                img = Image.open(os.path.join(img_dir,img)).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                else:
                    raise Exception("No transform")
                img_input.append(img)
            out = torch.stack(img_input,dim=0).cuda()
            if out.shape[0]>=self.num_samples:
                out = out[:self.num_samples,:]
                out = out.reshape(out.shape[0]*out.shape[1],*out.shape[2:])
            else:
                print(f"not enough samples for {path}")
                index = [j%out.shape[0] for j in range(self.num_samples)]
                out = out[index,:]
                out = out.reshape(out.shape[0]*out.shape[1],*out.shape[2:])
            img_features.append(out.detach().cpu())
        return img_features

    def get_kpts_features(self):
        kpts_dir = os.path.join(self.data_path,f'{self.phase}_set',"kpts_abs")

        kpts_names = os.listdir(kpts_dir)
        kpts_features = []
        for i in range(len(self.paths)):
            path = self.paths[i]
            cur_kpts = [x for x in kpts_names if path in x]
            if len(cur_kpts)<1:
                raise Exception(f"No kpts found in {path}")
            kpts_input = []
            for kpt in cur_kpts:
                kpts = np.load(os.path.join(kpts_dir,kpt))
                kpts= torch.tensor(kpts.flatten())*0.01
                kpts_input.append(kpts)
            out = torch.stack(kpts_input,dim=0).cuda()
            if out.shape[0]>=self.num_samples:
                out = out[:self.num_samples,:]
                out = out.flatten()
            else:
                print(f"not enough samples for {path}")
                index = [j%out.shape[0] for j in range(self.num_samples)]
                out = out[index,:]
                out = out.flatten()
            kpts_features.append(out.detach().cpu())
        return kpts_features
    
    def get_audio_features(self):
        audio_dir = os.path.join(self.data_path,f'{self.phase}_set',"audio")
        audios = os.listdir(audio_dir)
        audio_features = []
        for i in range(len(self.paths)):
            path = self.paths[i]
            for audio in audios:
                if audio[:-4] == path[5:-6]:
                    cores_path = audio 
            audio = torch.load(os.path.join(audio_dir,cores_path)).squeeze().detach().cpu()
            audio_features.append(audio)
        return audio_features
        
    def adjust_class(self,label, num_class):
        if num_class == 2:
            label[label>0]=1   
        elif num_class==6:
            label[(2>=label) *(label>0)]=1   
            label[(4>=label) *(label>2)]=2
            label[(6>=label) *(label>4)]=3 
            label[(8>=label) *(label>6)]=4
            label[(10>=label) *(label>8)]=5
        elif num_class ==3:
            label[(5>=label) *(label>0)]=1   
            label[(10>=label) *(label>5)]=2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):

        return {
            "img": self.img_features[idx],
            "kpt": self.kpts_features[idx],
            "audio": self.audio_features[idx],
            "label": self.label[idx],
            "path": self.paths[idx]
            }


def argmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)


def run_training():
    
    mods = ["img","kpt","audio"]
    num_samples = 8
    num_class = 6
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    torch.manual_seed(3)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    img_net = resnet50().cuda()
    # img_net.eval()
    for param in img_net.parameters():
        param.requires_grad = False

    model = combined_model(num_class = num_class,num_samples=num_samples,img_net=img_net).to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    
    train_dataset = PDDataSet(args.pd_path, phase = 'train', num_class = num_class,transform = data_transforms,im_net=img_net,num_samples=num_samples)   

    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

    val_dataset = PDDataSet(args.pd_path, phase = 'val', num_class = num_class, transform = data_transforms_val, im_net=img_net,num_samples=num_samples)   
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    print('Validation set size:', val_dataset.__len__())
    
    criterion_cls = torch.nn.CrossEntropyLoss()
    bce_cls = torch.nn.BCELoss()

    params = list(model.parameters()) #+ list(criterion_af.parameters())
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    soft_max = torch.nn.Softmax(dim=1).to(device)
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            
            out_dict, targets= model(data)#[targets.shape[0]:]
            iter_cnt += 1
            loss = 0
            final_out = 0
            for modal in mods:

                out = out_dict[modal]
                if modal in ["img","kpt"]:
                    train_targets = targets[:,None].repeat(1,num_samples).view(-1)
                    one_hot = torch.zeros_like(out).scatter_(1, torch.argmax(out, dim=1).unsqueeze(1), 1.)
                    video_out = one_hot.view(targets.shape[0],num_samples,-1).sum(dim=1)
                    video_out = video_out/num_samples
                    final_out += video_out
                else:
                    train_targets = targets
                    final_out += soft_max(out)
                loss += criterion_cls(out,train_targets)*0.1
                loss += bce_cls(1-soft_max(out)[:,0],(train_targets>0).float())*0.9
                
            final_out /= len(mods) 
            predicts = torch.argmax(final_out, dim=1)
            running_loss += loss.item()
            correct_sum += (predicts==targets).sum().item()

            # print(loss)
            loss.backward()
            optimizer.step()

        # print(val_pred)
        scheduler.step()
        acc = float(correct_sum) / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        print('Epoch: {}, Loss: {:.4f}, Acc: {:.4f}'.format(epoch, running_loss, acc))
        logits_all = []
        target_all=[]
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            correct_num = 0
            model.eval()
            
            for data in val_loader:
      
                # out_dict, targets = model(data)
                out_dict, targets= model(data)#[targets.shape[0]:]
                iter_cnt += 1
                loss = 0
                final_out = 0
                for modal in mods:
                    out = out_dict[modal]
                    if modal in ["img","kpt"]:
                        train_targets = targets[:,None].repeat(1,num_samples).view(-1)
                        one_hot = torch.zeros_like(out).scatter_(1, torch.argmax(out, dim=1).unsqueeze(1), 1.)
                        video_out = one_hot.view(targets.shape[0],num_samples,-1).sum(dim=1)
                        video_out = video_out/num_samples
                        final_out += video_out
                    else:
                        train_targets = targets
                        final_out += soft_max(out)
                    loss += criterion_cls(out,train_targets)
                    loss += bce_cls(1-soft_max(out)[:,0],(train_targets>0).float())
                # print(final_out[:3,:])
                running_loss += 0
                iter_cnt+=1
                logits_all.append(final_out)
                target_all.append(targets)
                # for m in range(final_out.shape[0]):
                #     print(data['path'][m])
                #     print(final_out[m,:])
                predicts = torch.argmax(final_out, dim=1)
                
                correct_num  += torch.eq(predicts,targets).sum().item()
            logits_all = torch.cat(logits_all, dim=0).detach().cpu()
            target_all = torch.cat(target_all, dim=0).detach().cpu()
            
            running_loss = running_loss/iter_cnt   
            acc = float(correct_num)  / float(val_dataset.__len__())
            final_f1, final_mf1, top1, final_auroc, total_mse, class_mse, class_recall, macro_mse, macro_recall = metrics(logits_all, target_all,num_class)
            print('Val Epoch: {}, Loss: {:.4f}, Acc: {:.4f}'.format(epoch, running_loss, acc))
            print(f'Multi Class... F1:{final_f1},Macro F1:{final_mf1},Top1:{top1},AUROC:{final_auroc}, Total MSE:{total_mse},  Macro MSE:{macro_mse}, Macro Recall:{macro_recall}')

    
    torch.save(model, "model.pt")

            


        
if __name__ == "__main__":        
    run_training()