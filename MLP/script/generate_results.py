import os
from pathlib import Path
import time
import pdb
from matplotlib.cm import get_cmap
import openpyxl
from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import *
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt
import pandas as pd
import copy
import argparse
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import DataLoader
from scipy.stats import entropy
from torch.utils.tensorboard import SummaryWriter

MLP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = MLP_ROOT / 'data'
MODEL_DIR = MLP_ROOT / 'models'
RESULTS_DIR = MLP_ROOT / 'results'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(MLP_ROOT / 'log'))
gene_dic={"P78":[0,0,0,0,0,0,0],"P74":[1,1,0,0,0,0,0],"P81":[1,0,0,1,0,0,0],"P112":[1,0,1,0,0,0,0],"P113":[0,0,1,1,0,0,0],"P114":[0,1,0,1,0,0,0],"P115":[0,0,0,0,0,1,0],"P116":[0,0,0,1,0,1,0],"P117":[0,1,0,0,0,1,0],"P118":[1,0,1,1,0,0,0],"P119":[1,1,1,0,0,0,0],"P120":[0,1,1,1,0,0,0],"P121":[1,1,0,1,0,0,0],"P122":[0,0,1,1,0,1,0],"P123":[1,0,0,1,0,1,0],"P124":[0,1,0,1,0,1,0],"P125":[0,1,1,0,0,0,0],"P126":[0,0,1,1,1,0,0],"P132":[1,0,0,1,1,0,0],"P135":[0,1,0,1,1,0,0],"P136":[1,1,0,0,1,0,0],"P140":[0,1,0,1,0,0,1],"P201":[1,1,1,1,0,0,0]}
# col_dic={"P74":0,"P81":1,"P118":2,"P201":3,"P119":4,"P120":5,"P121":6,"P124":7,"P135":8,"P136":9,"P140":10,"P78":11,"P114":12,"P116":13,"P117":14,"P122":15,"P123":16,"P125":17,"P126":18,"P132":19}
cluster_group={'P78':3,'P74':1,'P81':2,'P114':1,'P116':3,'P117':0,'P118':2,'P119':1,'P120':1,'P121':1,'P122':3,'P123':2,'P124':0,'P125':1,'P126':3,'P132':2,'P135':0,'P136':1,'P140':0,'P112':2,'P113':3}
name_dic={0:'Trp53',1:'Nf1',2:'Rb1',3:'Pten',4:'Pi3kcg',5:'Cdkn2a/2b',6:'EgfrvIII',7:'0',8:'1',9:'2',10:'3'}
subtype_2010_3_1={'Proneural':0,'Classical':1,'Mesenchymal':2,'Neural':3}
subtype_2017_3_1={'Proneural':0,'Classical':1,'Mesenchymal':2}
subtype_2010_1_1={'Proneural':0,'Classical':1,'Mesenchymal':2,'Neural':3}
subtype_2017_1_1={'Proneural':0,'Classical':1,'Mesenchymal':2}
my_one_hot={0:[1,0,0,0],1:[0,1,0,0],2:[0,0,1,0],3:[0,0,0,1]}
device=torch.device('cpu')
def load_mnist3_1():
    excel_file = DATA_DIR / 'molecular_phenotypes.xlsx'
    workbook=openpyxl.load_workbook(excel_file)
    sheet=workbook.active
    data_in=[]
    data_out=[]
    name=[]
    i=0
    for row in  sheet.iter_rows(values_only=True):
        if i>0:
            tmp_out=[]
            for j,k,in enumerate(row):
                # if j<4:
                #     print(j,k)
                if j==0:
                    mm,nn=k.split('_')
                    name.append(mm)
                if j==4:
                    tmp_out.append(int(k[-1])-1)
                    # print(int(k[-1])-1)
                if j==3:
                    tmp_out.append(subtype_2010_3_1[k])
                if j==2:
                    tmp_out.append(subtype_2017_3_1[k])
            tmptmp_out=[tmp_out[2],tmp_out[1],tmp_out[0]]
            data_out.append(tmptmp_out)
        i+=1
    for i in range(len(name)):
        if not name[i] in cluster_group:
            data_in.append(gene_dic[name[i]]+[0,0,0,0])
        else:
            data_in.append(gene_dic[name[i]]+my_one_hot[cluster_group[name[i]]])
    rel_data_in=[]
    rel_data_out=[]
    for i in range(len(name)):
        x=copy.deepcopy(data_in[i])
        # print(x)
        tot=0
        for j in range(11):
            if x[j]==1:
                tot+=1
        # print(tot)
        for j in range(2**tot):
            # print(j)
            tmp=copy.deepcopy(x)
            # print(tmp)
            tot2=tot-1
            now=j
            k=10
            while k>=0:
                if x[k]==1:
                    if now>=2**tot2:
                        now-=2**tot2
                        tmp[k]=0
                    tot2-=1
                k-=1
            # print(tmp)
            rel_data_in.append(tmp)
            rel_data_out.append(data_out[i])
    rel_data_in=torch.FloatTensor(rel_data_in)
    rel_data_out=torch.LongTensor(rel_data_out)
    # print(rel_data_in.size(0))
    return rel_data_in,rel_data_out
def normalize_list(input_list):
    total_sum = sum(input_list)
    if total_sum == 0:
        return [0] * len(input_list)  # Avoid division by zero.
    return [x / total_sum for x in input_list]
def load_mnist3_1_2():
    excel_file = DATA_DIR / 'microenvironment_phenotypes.xlsx'
    workbook=openpyxl.load_workbook(excel_file)
    sheet=workbook.active
    data_in=[]
    data_out=[]
    sums=[]
    name=[]
    i=0
    for row in sheet.iter_rows(values_only=True):
        if i>=1:
            tmp_out=[]
            for j,k,in enumerate(row):
                if j==0:
                    # print(k)
                    mm,nn=k.split('_')
                    name.append(mm)
                elif j>1:
                    # print(j,i,k)
                    tmp_out.append(k)
            sums.append(sum(tmp_out))
            data_out.append(normalize_list(tmp_out))
        i+=1
    for i in range(len(name)):
        if not name[i] in cluster_group:
            data_in.append(gene_dic[name[i]]+[0,0,0,0])
        else:
            data_in.append(gene_dic[name[i]]+my_one_hot[cluster_group[name[i]]])
    rel_data_in=[]
    rel_data_out=[]
    rel_sums=[]
    for i in range(len(name)):
        x=copy.deepcopy(data_in[i])
        # print(x)
        tot=0
        for j in range(11):
            if x[j]==1:
                tot+=1
        # print(tot)
        for j in range(2**tot):
            # print(j)
            tmp=copy.deepcopy(x)
            # print(tmp)
            tot2=tot-1
            now=j
            k=10
            while k>=0:
                if x[k]==1:
                    if now>=2**tot2:
                        now-=2**tot2
                        tmp[k]=0
                    tot2-=1
                k-=1
            # print(tmp)
            rel_data_in.append(tmp)
            rel_data_out.append(data_out[i])
            rel_sums.append(sums[i])
            # print(data_out[i])
    rel_data_in=torch.FloatTensor(rel_data_in)
    rel_data_out=torch.FloatTensor(rel_data_out)
    # print(rel_data_in.size(0))
    return rel_data_in,rel_data_out,rel_sums     
def load_mnist3_3():
    # the data, shuffled and split between train and test sets
    excel_file = DATA_DIR / 'survival_phenotypes.xlsx'
    workbook=openpyxl.load_workbook(excel_file)
    sheet=workbook.active
    # print(df.head())
    # print(df[0,1])
    data_in=[]
    data_out=[]
    name=[]
    i=0
    for row in sheet.iter_rows(values_only=True):
        # print(row)
        if i>0:
            tmp_out=[]
            a=0
            b=0
            p=1
            plkplk=""
            for j,k in enumerate(row):
                if j==0:
                    _, kk = k.split('-')
                    if kk in gene_dic:
                        plkplk=kk
                    else:
                        p=0
                        break
                    # print(kk)
                if j==13:
                    a=k
                    tmp_out.append(a/100)
                if j==14:
                    tmp_out.append(k/10)
            if p:
                name.append(plkplk)
                data_out.append(tmp_out)
                print(tmp_out)
        i+=1
    # data=torch.FloatTensor(data)
    # print(name)
    # print(len(data))
    # print(name)
    for i in range(len(name)):
        if not name[i] in cluster_group:
            data_in.append(gene_dic[name[i]]+[0,0,0,0])
        else:
            data_in.append(gene_dic[name[i]]+my_one_hot[cluster_group[name[i]]])
        #cluster_centers[cluster_group[name[i]]]
        # data[i]=torch.concat((data[i],torch.FloatTensor(cluster_centers[cluster_group[name[i]]])),dim=0)
    # print(data)
    # data=torch.FloatTensor(data)
    # mean_data=torch.mean(data,dim=0)
    # std_data=torch.std(data,dim=0)
    # data=(data-mean_data)/(std_data+1e-6)
    rel_data_in=[]
    rel_data_out=[]
    for i in range(len(name)):
        x=copy.deepcopy(data_in[i])
        # print(x)
        tot=0
        for j in range(11):
            if x[j]==1:
                tot+=1
        # print(tot)
        for j in range(2**tot):
            # print(j)
            tmp=copy.deepcopy(x)
            # print(tmp)
            tot2=tot-1
            now=j
            k=10
            while k>=0:
                if x[k]==1:
                    if now>=2**tot2:
                        now-=2**tot2
                        tmp[k]=0
                    tot2-=1
                k-=1
            # print(tmp)
            rel_data_in.append(tmp)
            rel_data_out.append(data_out[i])
    rel_data_in=torch.FloatTensor(rel_data_in)
    rel_data_out=torch.FloatTensor(rel_data_out)
    return rel_data_in,rel_data_out
class my_model(nn.Module):
    def __init__(self):
        super(my_model,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(11,8),
            nn.ReLU(True))
        self.model1_1=model_1_1()
        self.model3_1=model_3_1()
        self.model3_1_2=model_3_1_2()
        self.model3_2=model_3_2()
        self.model3_3=model_3_3()
    def encode(self, x):
        return self.encoder(x)
    def forward1_1(self, x):
        fet=self.encode(x)
        return self.model1_1(fet)
    def forward3_1(self, x):
        fet=self.encode(x)
        return self.model3_1(fet)
    def forward3_1_2(self, x):
        fet=self.encode(x)
        return self.model3_1_2(fet)
    def forward3_2(self, x):
        fet=self.encode(x)
        return self.model3_2(fet)
    def forward3_3(self, x):
        fet=self.encode(x)
        return self.model3_3(fet)
    def forward(self, x):
        fet=self.encode(x)
        return self.model3_1(fet),self.model3_2(fet),self.model3_3(fet)
class model_1_1(nn.Module):
    def __init__(self):
        super(model_1_1,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(8,6),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(#cluster_group_3
            nn.Linear(6,3),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(#cluster_group_4
            nn.Linear(6,4),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(#subtype_2010
            nn.Linear(6,4),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(#subtype_2017
            nn.Linear(6,4),
            nn.ReLU(True))
    def forward(self, x):
        fet=self.layer1(x)
        out1=self.layer2(fet)
        out2=self.layer3(fet)
        out3=self.layer4(fet)
        out4=self.layer5(fet)
        return out1,out2,out3,out4
class model_3_1(nn.Module):
    def __init__(self):
        super(model_3_1,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(8,6),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(#cluster_group
            nn.Linear(6,2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(#subtype_2010
            nn.Linear(6,4),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(#subtype_2017
            nn.Linear(6,4),
            nn.ReLU(True))
    def forward(self, x):
        fet=self.layer1(x)
        out1=self.layer2(fet)
        out2=self.layer3(fet)
        out3=self.layer4(fet)
        return out1,out2,out3
class model_3_1_2(nn.Module):
    def __init__(self):
        super(model_3_1_2,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(8,10),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(10,13),
            nn.ReLU(True))
    def forward(self, x):
        fet=self.layer1(x)
        out=self.layer2(fet)
        return out
class model_3_2(nn.Module):
    def __init__(self):
        super(model_3_2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(8, 14),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(14, 19),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(14, 19),
            nn.ReLU(True))
    def forward(self, x):
        fet=self.layer1(x)
        out1=self.layer2(fet)
        out2=self.layer3(fet)
        return out1,out2
class model_3_3(nn.Module):
    def __init__(self):
        super(model_3_3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(4,1),
            nn.Sigmoid()
            )
        self.layer3 = nn.Sequential(
            nn.Linear(4,1),
            nn.ReLU(True)
        )
    def forward(self, x):
        fet=self.layer1(x)
        out1=self.layer2(fet)
        out2=self.layer3(fet)
        return out1,out2
class MyDataSet_1_1(Dataset):
    def __init__(self,data,label_1_1_1,label_1_1_2,label_1_1_3,label_1_1_4):
        self.data=data
        self.label1_1_1=label_1_1_1
        self.label1_1_2=label_1_1_2
        self.label1_1_3=label_1_1_3
        self.label1_1_4=label_1_1_4
    def __getitem__(self, index):
        return (self.data[index],self.label1_1_1[index],self.label1_1_2[index],self.label1_1_3[index],self.label1_1_4[index])
    def __len__(self):
        return len(self.data)
class MyDataSet_3_1(Dataset):
    def __init__(self,data,label_3_1_1,label_3_1_2,label_3_1_3):
        self.data=data
        self.label3_1_1=label_3_1_1
        self.label3_1_2=label_3_1_2
        self.label3_1_3=label_3_1_3
    def __getitem__(self, index):
        return (self.data[index],self.label3_1_1[index],self.label3_1_2[index],self.label3_1_3[index])
    def __len__(self):
        return len(self.data)
class MyDataSet_3_1_2(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __getitem__(self, index):
        return (self.data[index],self.label[index])
    def __len__(self):
        return len(self.data)
class MyDataSet_3_2(Dataset):
    def __init__(self,data,label3_2_1,label3_2_2):
        self.data=data
        self.label3_2_1=label3_2_1
        self.label3_2_2=label3_2_2
    def __getitem__(self, index):
        return (self.data[index],self.label3_2_1[index],self.label3_2_2[index])
    def __len__(self):
        return len(self.data)
class MyDataSet_3_3(Dataset):
    def __init__(self,data,label3_3):
        self.data=data
        self.label3_3=label3_3
    def __getitem__(self, index):
        return (self.data[index],self.label3_3[index])
    def __len__(self):
        return len(self.data)

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")
def train(**kwargs):
    data3_1,label3_1,data3_1_2,label3_1_2,data3_3,label3_3=kwargs['data']
    model=kwargs['model']
    n_epoches=kwargs['num_epochs']
    batch_size=kwargs['batch_size']
    savepath=kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    train_data3_1=MyDataSet_3_1(data3_1,label3_1[:,0],label3_1[:,1],label3_1[:,2])
    train_data3_1_2=MyDataSet_3_1_2(data3_1_2,label3_1_2)
    train_data3_3=MyDataSet_3_3(data3_3,label3_3)
    train_loader3_1=DataLoader(dataset=train_data3_1,
                            batch_size=batch_size,
                            shuffle=True)
    train_loader3_1_2=DataLoader(dataset=train_data3_1_2,
                            batch_size=batch_size,
                            shuffle=True)
    train_loader3_3=DataLoader(dataset=train_data3_3,
                            batch_size=batch_size,
                            shuffle=True)
    
    model=model.to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=6e-3, weight_decay=1e-5)
    loss_compute1_1=nn.CrossEntropyLoss(reduction='sum')
    loss_compute3_1=nn.CrossEntropyLoss(reduction='sum')
    loss_compute3_1_2=nn.KLDivLoss(reduction='sum')
    loss_compute3_2=nn.MSELoss(reduction='sum')
    loss_compute3_3_1=nn.BCELoss(reduction='sum')
    loss_compute3_3_2=nn.MSELoss(reduction='sum')
    for epoch in range(start_epoch,n_epoches):
        train_loader3_1_2_iterator=iter(train_loader3_1_2)
        train_loader3_3_iterator=iter(train_loader3_3)
        tot1_1=0
        tot3_1=0
        tot3_1_2=0
        tot3_2=0
        tot3_3=0
        tot_loss1_1=0
        tot_loss3_1=0
        tot_loss3_1_2=0
        tot_loss3_2=0
        tot_loss3_3_1=0
        tot_loss3_3_2=0
        k3_1=1.5
        k3_1_2=7.5
        k3_3_1=2
        k3_3_2=0.003
        for data3_1 in train_loader3_1:
            
            #train 3-1
            img,lbl1,lbl2,lbl3=data3_1
            img=img.float().to(device)
            lbl1=lbl1.long().to(device)
            lbl2=lbl2.long().to(device)
            out1,out2,out3=model.forward3_1(img)
            # print(out1,lbl1)
            # print(out1,lbl1)
            loss1=loss_compute3_1(out1,lbl1)
            loss2=loss_compute3_1(out2,lbl2)
            loss=loss1+loss2
            tot3_1+=img.size(0)
            loss*=k3_1
            tot_loss3_1+=loss
            loss/=img.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            
            #train 3-1-2
            try:
                data3_1_2=next(train_loader3_1_2_iterator)
            except StopIteration:
                train_loader3_1_2_iterator=iter(train_loader3_1_2)
                data3_1_2=next(train_loader3_1_2_iterator)
            img,lbl=data3_1_2
            img=img.float().to(device)
            lbl=lbl.float().to(device)
            out=model.forward3_1_2(img)
            loss=loss_compute3_1_2(out.softmax(-1).log(),lbl)
            tot3_1_2+=img.size(0)
            loss*=k3_1_2
            tot_loss3_1_2+=loss
            loss/=img.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
             
            
            #train 3-3
            try:
                data3_3=next(train_loader3_3_iterator)
            except StopIteration:
                train_loader3_3_iterator=iter(train_loader3_3)
                data3_3=next(train_loader3_3_iterator)
            img,lbl=data3_3
            img=img.float().to(device)
            lbl=lbl.float().to(device)
            out3_3_1,out3_3_2=model.forward3_3(img)
            out3_3_1=out3_3_1.squeeze(-1)
            out3_3_2=out3_3_2.squeeze(-1)
            lbl3_3_1=lbl[:,0]
            lbl3_3_2=lbl[:,1]
            loss3_3_1=loss_compute3_3_1(out3_3_1,lbl3_3_1)
            # print(out3_3_1,lbl3_3_1)
            # print(out3_3_2.shape,lbl3_3_2.shape)
            # print(out3_3_2,lbl3_3_2)
            loss3_3_2=loss_compute3_3_2(out3_3_2,lbl3_3_2)
            loss3_3_1*=k3_3_1
            loss3_3_2*=k3_3_2
            # print(loss3_3_1,loss3_3_2)
            loss=loss3_3_1+loss3_3_2
            tot3_3+=img.size(0)
            tot_loss3_3_1+=loss3_3_1
            tot_loss3_3_2+=loss3_3_2
            loss/=img.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], 3-1_loss:{:.4f}'
          .format(epoch + 1, n_epoches, tot_loss3_1.item()/tot3_1))
        print('epoch [{}/{}], 3-1-2_loss:{:.4f}'
          .format(epoch + 1, n_epoches, tot_loss3_1_2.item()/tot3_1_2))
        print('epoch [{}/{}], 3-3-1_loss:{:.4f}'
          .format(epoch + 1, n_epoches, tot_loss3_3_1.item()/tot3_3))
        print('epoch [{}/{}], 3-3-2_loss:{:.4f}'
          .format(epoch + 1, n_epoches, tot_loss3_3_2.item()/tot3_3))
        # writer.add_scalar('1-1_loss',tot_loss1_1.item()/tot1_1,epoch)
        writer.add_scalar('3-1_loss',tot_loss3_1.item()/tot3_1,epoch)
        writer.add_scalar('3-1-2_loss',tot_loss3_1_2.item()/tot3_1_2,epoch)
        writer.add_scalar('3-3-1_loss',tot_loss3_3_1.item()/tot3_3,epoch)
        writer.add_scalar('3-3-2_loss',tot_loss3_3_2.item()/tot3_3,epoch)
        # print(tot_loss3_2)
        state = (tot_loss3_1.item()/tot3_1) + (tot_loss3_1_2.item()/tot3_1_2) + (tot_loss3_3_1.item()/tot3_3) + (tot_loss3_3_2.item()/tot3_3)
        # print(state)
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best': state,
                        'epoch':epoch
                        }, savepath,
                        is_best)
            
    print(checkpoint['best'])
parser = argparse.ArgumentParser(description='MLP phenotype score generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train', action='store_true', help='Train or resume fitting before generating scores')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--train_epochs', default=3200, type=int)
parser.add_argument('--save_dir', default='saves')
args = parser.parse_args()
data_in_3_1,data_out_3_1=load_mnist3_1()
data_in_3_1_2,data_out_3_1_2,sums_3_1_2=load_mnist3_1_2()
data_in_3_3,data_out_3_3=load_mnist3_3()
save_path = MODEL_DIR / 'trained_model.pth'
model=my_model()
if os.path.isfile(save_path):
    print('Loading {}'.format(save_path))
    checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
else:
    if not args.train:
        parser.error('Main checkpoint not found. Restore models/trained_model.pth or use --train to fit a new model.')
    print("=> no checkpoint found at '{}'".format(save_path))
    checkpoint = {
        "epoch": 0,
        "best": float("inf")
    }


if args.train:
    if args.train_epochs < 1:
        parser.error('--train_epochs must be positive when --train is used')
    train(data=(data_in_3_1,data_out_3_1,data_in_3_1_2,data_out_3_1_2,data_in_3_3,data_out_3_3),model=model,batch_size=args.batch_size,num_epochs=args.train_epochs,savepath=save_path, checkpoint=checkpoint)

checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
dic={}
# dic1_1={}
dic3_1={}
# dic3_2={}
dic3_3={}
dic3_1_2={}
# for i in range(data_in_1_1.size(0)):
#     # tmp=""
#     # for k in range(data_in_1_1[i].size(0)):
#     #     if data_in_1_1[i,k]==1:
#     #         tmp=tmp+name_dic[k]+', '
#     x=tuple(data_in_1_1[i].tolist())
#     if not x in dic1_1:
#         # print(tmp)
#         dic1_1.update({x:1})
#     else:
#         dic1_1[x]+=1
for i in range(data_in_3_1.size(0)):
    x=tuple(data_in_3_1[i].tolist())
    if not x in dic3_1:
        dic3_1.update({x:1})
    else:
        dic3_1[x]+=1
for i in range(data_in_3_1_2.size(0)):
    x=tuple(data_in_3_1_2[i].tolist())
    if not x in dic3_1_2:
        dic3_1_2.update({x:1})
    else:
        dic3_1_2[x]+=1
# for i in range(data_in_3_2.size(0)):
#     x=tuple(data_in_3_2[i].tolist())
#     if not x in dic3_2:
#         dic3_2.update({x:1})
#     else:
#         dic3_2[x]+=1
for i in range(data_in_3_3.size(0)):
    x=tuple(data_in_3_3[i].tolist())
    if not x in dic3_3:
        dic3_3.update({x:1})
    else:
        dic3_3[x]+=1
for i in range(data_in_3_3.size(0)):
    x=tuple(data_in_3_3[i].tolist())
    p1=1
    p4=1
    for j in range(data_in_3_1.size(0)):
        y=tuple(data_in_3_1[j].tolist())
        # print(x,y)
        if x==y:
            p1=0
            break
    for j in range(data_in_3_1_2.size(0)):
        y=tuple(data_in_3_1_2[j].tolist())
        if x==y:
            p4=0
            break
    if p1 or p4:
    # if p1 or p2:
        continue
    if not x in dic:
        dic.update({x:1})
    else:
        dic[x]+=1
# for i,j in enumerate(dic):
#     tmp=""
#     x=torch.FloatTensor(list(j)).to(device)
#     for k in range(x.size(0)):
#         if x[k]==1:
#             tmp=tmp+name_dic[k]+', '
#     if tmp=='':
#         continue
#     if dic3_1[j]<=3 or dic3_1_2[j]<=1 or dic3_2[j]<1 or dic3_3[j]<1:
#         continue
#     print(tmp)
# # print(dic)
##############
#find all the samples of a set
# lis=[]
# for i,j in enumerate(dic):
#     x=torch.FloatTensor(list(j)).to(device)
#     tmp=""
#     for k in range(x.size(0)):
#         if x[k]==1:
#             tmp=tmp+name_dic[k]+', '
#     if tmp=='' or x[4]==1 or x[5]==1:# Exclude Pi3kcg and Cdkn2a/Cdkn2b.
#         continue
#     if dic3_1[j]<=1 or (dic3_1_2[j]<=1 and dic3_2[j]<=1 and dic3_3[j]<=1):
#         continue
#     tmp=tmp[:-2]
#     tmpp=[]
#     # print(j[0])
#     for A,B in enumerate(gene_dic):
#         p=1
#         for k in range(7):
#             if j[k]==1 and gene_dic[B][k]==0:
#                 p=0
#                 break
#         for k in range(7,11):
#             if j[k]==1 and ((not B in cluster_group) or (B in cluster_group and cluster_group[B]!=k-7)):
#                 p=0
#                 break
#         if p:
#             tmpp.append(B)
#     lis.append((tmp,tmpp))
# print(lis)
# wb = openpyxl.Workbook()
# sheet=wb.active
# for i in range(len(lis)):
#     A,B=lis[i]
#     B=tuple(B)
#     sheet.cell(row=i+1,column=1).value=A
#     for j in range(len(B)):
#         sheet.cell(row=i+1,column=j+2).value=B[j]
# wb.save('samples.xlsx')
#######################
#####################
#count all the data appearance
dic_count={}
for i in range(data_in_3_1.size(0)):
    x=tuple(data_in_3_1[i].tolist())
    y=torch.FloatTensor(list(x)).to(device)
    tmp=""
    for k in range(y.size(0)):
        if y[k]==1:
            tmp=tmp+name_dic[k]+', '
    if tmp=='':
        continue
    tmp=tmp[:-2]
    if not tmp in dic_count:
        dic_count.update({tmp:1})
    else:
        dic_count[tmp]+=1
for i in range(data_in_3_1_2.size(0)):
    x=tuple(data_in_3_1_2[i].tolist())
    y=torch.FloatTensor(list(x)).to(device)
    tmp=""
    for k in range(y.size(0)):
        if y[k]==1:
            tmp=tmp+name_dic[k]+', '
    if tmp=='':
        continue
    tmp=tmp[:-2]
    if not tmp in dic_count:
        dic_count.update({tmp:1})
    else:
        dic_count[tmp]+=1
for i in range(data_in_3_3.size(0)):
    x=tuple(data_in_3_3[i].tolist())
    y=torch.FloatTensor(list(x)).to(device)
    tmp=""
    for k in range(y.size(0)):
        if y[k]==1:
            tmp=tmp+name_dic[k]+', '
    if tmp=='':
        continue
    tmp=tmp[:-2]
    if not tmp in dic_count:
        dic_count.update({tmp:1})
    else:
        dic_count[tmp]+=1
print(dic_count)
# ##############################
# #Calculate the Manhattan distances of the samples.
# out2_2=[]
# for i,j in enumerate(dic):
#     x=torch.FloatTensor(list(j)).to(device)
#     tmp=""
#     for k in range(x.size(0)):
#         if x[k]==1:
#             tmp=tmp+name_dic[k]+', '
#     if tmp=='':
#         continue
#     if dic3_1[j]<=3 or dic3_1_2[j]<=1 or dic3_2[j]<1 or dic3_3[j]<1:
#         continue
#     tmp=tmp[:-2]
#     # print(tmp)
#     out21,out22=model.forward3_2(x)
#     out21/=100
#     out22/=100
#     out21=out21.tolist()
#     out22=out22.tolist()
#     for k in range(len(out21)):
#         if out21[k]<0.1:
#             out21[k]=0
#     for k in range(len(out22)):
#         if out22[k]<0.1:
#             out22[k]=0
#     out=out21+out22
#     out2_2.append(out)
#     # print(out)
# out2_2=torch.FloatTensor(out2_2)
# final_out2_2=torch.zeros(out2_2.size(0))
# for i in range(out2_2.size(0)):
#     sum=0
#     for j in range(out2_2.size(0)):
#         if i==j:
#             continue
#         sum+=torch.dist(out2_2[i],out2_2[j],p=1)
#     sum/=(out2_2.size(0)-1)
#     final_out2_2[i]=copy.deepcopy(sum.detach())
# # print(final_out2_2)
# final_out2_2=final_out2_2.tolist()
# dic_final_out2_2={}
# for i,j in enumerate(dic):
#     x=torch.FloatTensor(list(j)).to(device)
#     tmp=""
#     for k in range(x.size(0)):
#         if x[k]==1:
#             tmp=tmp+name_dic[k]+', '
#     if tmp=='':
#         continue
#     if dic3_1[j]<=3 or dic3_1_2[j]<=1 or dic3_2[j]<1 or dic3_3[j]<1:
#         continue
#     tmp=tmp[:-2]
#     # print(i)
#     dic_final_out2_2.update({tmp:final_out2_2[i-1]})
# # ########################


###########################
#find features for 3-1-2
dic_sums={}
for i in range(data_in_3_1_2.size(0)):
    x=tuple(data_in_3_1_2[i].tolist())
    y=torch.FloatTensor(list(x)).to(device)
    tmp=""
    for k in range(y.size(0)):
        if y[k]==1:
            tmp=tmp+name_dic[k]+', '
    if tmp=='':
        continue
    tmp=tmp[:-2]
    dic_sums.update({tmp:sums_3_1_2[i]})
print(dic_sums)
out3_1_2=[]
dic_final_out3_1_2={}
tot_3_1_2_2_2=[]
for i,j in enumerate(dic):
    x=torch.FloatTensor(list(j)).to(device)
    tmp=""
    for k in range(x.size(0)):
        if x[k]==1:
            tmp=tmp+name_dic[k]+', '
    if tmp=='':
        continue
    if dic3_1[j]<=3 or dic3_1_2[j]<=1  or dic3_3[j]<1:
        continue
    # if tmp=='' or x[4]==1 or x[5]==1:# Exclude Pi3kcg and Cdkn2a/Cdkn2b.
    #     continue
    # if dic3_1[j]<=1 or (dic3_1_2[j]<=1 and dic3_2[j]<=1 and dic3_3[j]<=1):
    #     continue
    tmp=tmp[:-2]
    out3_1_2=model.forward3_1_2(x)
    out3_1_2=out3_1_2.softmax(-1)
    out3_1_2=out3_1_2.tolist()
    # out3_1_2_2_2=torch.FloatTensor([out3_1_2[3]/dic_sums[tmp]])
    out3_1_2_2_2=torch.FloatTensor([out3_1_2[12]])
    tot_3_1_2_2_2.append(out3_1_2_2_2)
# print(maxn)
sorted_tot_3_1_2_2_2=sorted(tot_3_1_2_2_2)
# print(len(sorted_tot_3_1_2_1))
threshold_3_1_2_2_2=0.28
threshold_3_1_2_2_1=0.15
now=0
for i,j in enumerate(dic):
    x=torch.FloatTensor(list(j)).to(device)
    tmp=""
    for k in range(x.size(0)):
        if x[k]==1:
            tmp=tmp+name_dic[k]+', '
    if tmp=='':
        continue
    # if tmp=='' or x[4]==1 or x[5]==1:# Exclude Pi3kcg and Cdkn2a/Cdkn2b.
    #     continue
    # if dic3_1[j]<=1 or (dic3_1_2[j]<=1 and dic3_2[j]<=1 and dic3_3[j]<=1):
    #     continue
    if dic3_1[j]<=3 or dic3_1_2[j]<=1  or dic3_3[j]<1:
        continue
    tmp=tmp[:-2]
    out3_1_2=model.forward3_1_2(x)
    out3_1_2=out3_1_2.softmax(-1)
    out3_1_2=out3_1_2.tolist()
    out3_1_2_2_1=torch.FloatTensor([out3_1_2[0]+out3_1_2[3]+out3_1_2[6]+out3_1_2[8]+out3_1_2[9]+out3_1_2[10]+out3_1_2[11],out3_1_2[1]+out3_1_2[5], out3_1_2[2]+out3_1_2[4]+out3_1_2[7]])# Immune cells: Myeloid (Microglia, Macrophages, monocytes), T cells, NK, mast cell, B cell, DC; glial cells: Astrocytes, Oligodendrocytes; vascular and epithelial cells: Endothelial cells, Pericytes, Ependymal cells.
    out3_1_2_2_1=out3_1_2_2_1/torch.sum(out3_1_2_2_1)
    #80~100:1, 20~80:0.5, 0~20:0
    # print(i-1,tot_3_1_2_1[i-1])


    feature3_1_2_2_2=0
    if sorted_tot_3_1_2_2_2.index(tot_3_1_2_2_2[now])>=(1-threshold_3_1_2_2_2)*len(tot_3_1_2_2_2):
        feature3_1_2_2_2=1
    # elif sorted_tot_3_1_2_2_2.index(tot_3_1_2_2_2[now])>=(1-2*threshold_3_1_2_2_2)*len(tot_3_1_2_2_2):
    elif sorted_tot_3_1_2_2_2.index(tot_3_1_2_2_2[now])>=(threshold_3_1_2_2_2)*len(tot_3_1_2_2_2):
        feature3_1_2_2_2=0.5
        
    feature3_1_2_2_1=-entropy(out3_1_2_2_1)
    maxn=torch.max(out3_1_2_2_1)
    lbl=[]
    if out3_1_2_2_1[0]>=maxn-threshold_3_1_2_2_1:
        lbl.append('immune')
    if out3_1_2_2_1[1]>=maxn-threshold_3_1_2_2_1:
        lbl.append('glial')
    if out3_1_2_2_1[2]>=maxn-threshold_3_1_2_2_1:
        lbl.append('vascular-epithelial')
    # print(maxn)
    dic_final_out3_1_2.update({tmp:(feature3_1_2_2_1,feature3_1_2_2_2,lbl)})
    
    # dic_final_out3_1_2.update({tmp:(feature3_1_2_2_1,tot_3_1_2_2_2[now].cpu().numpy()[0],lbl)})
    now+=1
# print(dic_final_out3_1_2)
    
###########################

# ##########################
# #find the cluster groups of out3_1_2
# out3_1_2=[]
# for i,j in enumerate(dic):
#     x=torch.FloatTensor(list(j)).to(device)
#     tmp=""
#     for k in range(x.size(0)):
#         if x[k]==1:
#             tmp=tmp+name_dic[k]+', '
#     if tmp=='' or x[4]==1 or x[5]==1:
#         continue
#     if dic3_1[j]<=1 or (dic3_1_2[j]<=1 and dic3_2[j]<=1 and dic3_3[j]<=1):
# #         p+=1
#         continue
#     tmp=tmp[:-2]
#     # print(tmp)
#     out=model.forward3_1_2(x)
#     out=out.softmax(-1)
#     out=out.tolist()
#     out3_1_2.append(out)
# # print(out3_1_2)
# out3_1_2=torch.FloatTensor(out3_1_2)
# kmeans=KMeans(n_clusters=4)
# kmeans.fit(out3_1_2)
# cls=kmeans.fit_predict(out3_1_2)
# t_sne=TSNE()
# x=copy.deepcopy(out3_1_2.detach())
# out3_1_2=out3_1_2.numpy()
# fig = plt.figure()
# ax=plt.subplot(111)
# x_embedded = TSNE(n_components=2).fit_transform(out3_1_2)

# colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'lime', 'pink',
# 				  'brown', 'teal', 'navy', 'olive', 'maroon', 'gold', 'indigo', 'salmon', 'peru', 'orchid']
# for i in range(x.size(0)):
#     plt.scatter(x_embedded[i,0],x_embedded[i,1],c=colors[cls[i]])
# plt.savefig('3_1_2.png')
# plt.close(fig)

# centers=kmeans.cluster_centers_
# centers=torch.FloatTensor(centers)
# print(centers)
# final_out3_1_2=[]
# for i in range(x.size(0)):
#     final_out3_1_2.append(torch.dist(x[i],centers[cls[i]]))
#     # print(final_out3_1_2[i],cls[i])
# dic_final_out3_1_2={}
# p=0
# for i,j in enumerate(dic):
#     y=torch.FloatTensor(list(j)).to(device)
#     tmp=""
#     for k in range(y.size(0)):
#         if y[k]==1:
#             tmp=tmp+name_dic[k]+', '
#     if tmp=='' or y[4]==1 or y[5]==1:
#         p+=1
#         continue
#     if dic3_1[j]<=1 or (dic3_1_2[j]<=1 and dic3_2[j]<=1 and dic3_3[j]<=1):
#         p+=1
#         continue
#     tmp=tmp[:-2]
#     dic_final_out3_1_2.update({tmp:final_out3_1_2[i-p-1]})
#     # print(tmp,':',cls[i-p-1])
# print(dic_final_out3_1_2)

##########################


wb = openpyxl.Workbook()
sheet=wb.active
p=0
nam=[]
for i,j in enumerate(dic):
    x=torch.FloatTensor(list(j)).to(device)
    tmp=""
    for k in range(x.size(0)):
        if x[k]==1:
            tmp=tmp+name_dic[k]+', '
    if tmp=="":
        p+=1
        continue
    if dic3_1[j]<=3 or dic3_1_2[j]<=1 or dic3_3[j]<1:
        p+=1
        continue
    tmp=tmp[:-2]
    nam.append(tmp)
    
    sheet.cell(row=i-p+1,column=1).value=tmp
    now=2    
    out11,out12,out13=model.forward3_1(x)
    out11=nn.functional.softmax(out11,dim=0)
    out12=nn.functional.softmax(out12,dim=0)
    out13=nn.functional.softmax(out13,dim=0)
    out11=out11.tolist()
    out12=out12.tolist()
    out13=out13.tolist()
    # print(tmp,out11)
    entropy11=entropy(out11)
    entropy12=entropy(out12)
    entropy13=entropy(out13)
    #=============
    #process 
    for ite in range(1):#12*3=36
        # for jj in range(len(out11)):
        #     sheet.cell(row=i-p+1,column=now).value=out11[jj]
        #     now+=1
        sheet.cell(row=i-p+1,column=now).value=-entropy11
        now+=1
    for ite in range(1):#12*3=36
        # for jj in range(len(out12)):
        #     sheet.cell(row=i-p+1,column=now).value=out12[jj]
        #     now+=1
        sheet.cell(row=i-p+1,column=now).value=-entropy12
        now+=1
        
    out3_1_2=model.forward3_1_2(x)
    out3_1_2=out3_1_2.softmax(-1)
    # print(out3_1_2.shape)
    out3_1_2=out3_1_2.tolist()
    A,B,C=dic_final_out3_1_2[tmp]
    #3*10=30
    # for ii in range(len(out3_1_2)):
        # sheet.cell(row=i-p+1,column=now).value=out3_1_2[ii]
        # now+=1
    for ite in range(1):
        sheet.cell(row=i-p+1,column=now).value=A
        now+=1
    for ite in range(1):
        sheet.cell(row=i-p+1,column=now).value=B
        now+=1
            
    
    out3_3_1,out3_3_2=model.forward3_3(x)
    out3_3_2*=10
    # print(tmp,out31)
    out3_3_1=out3_3_1.tolist()
    out3_3_2=out3_3_2.tolist()
    sheet.cell(row=i-p+1,column=now).value=out3_3_1[0]
    now+=1
    sheet.cell(row=i-p+1,column=now).value=out3_3_2[0]
    now+=1
    
    # sheet.cell(row=i-p+1,column=now).value=out3_1_2[3]/dic_sums[tmp]
    # for k in range(2):
    #     for ite in range(1):#38
    #         if k==0:
    #             sheet.cell(row=i-p+1,column=now).value=out3_3_1[k]
    #             now+=1
    #         if k==1:
    #             sheet.cell(row=i-p+1,column=now).value=out3_3_2[k-1]
    #             now+=1
print(nam)
wb.save(RESULTS_DIR / 'predicted_scores.xlsx')
# wb.save('raw_scores.xlsx')
