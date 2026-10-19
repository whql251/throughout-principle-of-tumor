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
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import copy
import argparse
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# excel_file = './result_v4_simplified_Manhattan_1-1_free.xlsx'
# excel_file='result_v4_simplified_Manhattan_1-1_2010_own_free.xlsx'
MLP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = MLP_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
excel_file = RESULTS_DIR / 'predicted_scores.xlsx'
workbook=openpyxl.load_workbook(excel_file)
sheet=workbook.active
wb=openpyxl.Workbook()
sht=wb.active
data=[]
nam=[]
now=0
for col in sheet.iter_cols(values_only=True):
    if now==0:
        for i,j in enumerate(col):
            nam.append(j)
    else:
        tmp=[]
        for i,j in enumerate(col):
            if j!=None:
                tmp.append(j)
            else:
                tmp.append(0)
        data.append(tmp)
    now+=1
data=torch.FloatTensor(data)

##########################
#count the num of all genes
# dic_count={'Nf1, Pten, 1': 60, 'Pten, 1': 60, 'Nf1, 1': 111, '1': 111, 'Nf1, Pten': 92, 'Pten': 124, 'Nf1': 144, 'Trp53, Nf1, Pten, 1': 33, 'Trp53, Pten, 1': 33, 'Trp53, Nf1, 1': 83, 'Trp53, 1': 83, 'Trp53, Nf1, Pten': 43, 'Trp53, Pten': 71, 'Trp53, Nf1': 93, 'Trp53': 122, 'Trp53, Pten, Cdkn2a/2b, 2': 8, 'Pten, Cdkn2a/2b, 2': 8, 'Trp53, Cdkn2a/2b, 2': 8, 'Cdkn2a/2b, 2': 8, 'Trp53, Pten, 2': 28, 'Pten, 2': 28, 'Trp53, 2': 29, '2': 29, 'Trp53, Pten, Cdkn2a/2b': 8, 'Pten, Cdkn2a/2b': 23, 'Trp53, Cdkn2a/2b': 8, 'Cdkn2a/2b': 25, 'Nf1, Pten, Cdkn2a/2b, 0': 14, 'Pten, Cdkn2a/2b, 0': 14, 'Nf1, Cdkn2a/2b, 0': 15, 'Cdkn2a/2b, 0': 15, 'Nf1, Pten, 0': 22, 'Pten, 0': 22, 'Nf1, 0': 23, '0': 23, 'Nf1, Pten, Cdkn2a/2b': 14, 'Nf1, Cdkn2a/2b': 15, 'Trp53, Nf1, Pi3kcg, 1': 7, 'Nf1, Pi3kcg, 1': 7, 'Trp53, Pi3kcg, 1': 7, 'Pi3kcg, 1': 7, 'Trp53, Nf1, Pi3kcg': 7, 'Nf1, Pi3kcg': 11, 'Trp53, Pi3kcg': 9, 'Pi3kcg': 13, '3': 9, 'Rb1, Pten, 3': 3, 'Pten, 3': 4, 'Rb1, 3': 3, 'Rb1, Pten': 45, 'Rb1': 68, 'Trp53, Rb1, Pten, 2': 10, 'Rb1, Pten, 2': 10, 'Trp53, Rb1, 2': 11, 'Rb1, 2': 11, 'Trp53, Rb1, Pten': 20, 'Trp53, Rb1': 42, 'Trp53, Nf1, Rb1, 1': 21, 'Nf1, Rb1, 1': 44, 'Trp53, Rb1, 1': 21, 'Rb1, 1': 44, 'Trp53, Nf1, Rb1': 31, 'Nf1, Rb1': 54, 'Nf1, Rb1, Pten, 1': 22, 'Rb1, Pten, 1': 22, 'Nf1, Rb1, Pten': 32, 'Trp53, Pten, Pi3kcg, 2': 2, 'Pten, Pi3kcg, 2': 2, 'Trp53, Pi3kcg, 2': 2, 'Pi3kcg, 2': 2, 'Trp53, Pten, Pi3kcg': 2, 'Pten, Pi3kcg': 6, 'Nf1, Pten, Pi3kcg, 0': 4, 'Pten, Pi3kcg, 0': 4, 'Nf1, Pi3kcg, 0': 4, 'Pi3kcg, 0': 4, 'Nf1, Pten, Pi3kcg': 4, 'Nf1, Pten, EgfrvIII, 0': 4, 'Pten, EgfrvIII, 0': 4, 'Nf1, EgfrvIII, 0': 4, 'EgfrvIII, 0': 4, 'Nf1, Pten, EgfrvIII': 4, 'Pten, EgfrvIII': 4, 'Nf1, EgfrvIII': 4, 'EgfrvIII': 4, 'Trp53, Nf1, Rb1, Pten': 10, 'Pten, Cdkn2a/2b, 3': 1, 'Cdkn2a/2b, 3': 1}
# # for i,j in enumerate(dic_count):
# #     print(j,':',dic_count[j])
# all_count=[]
# for i in range(len(nam)):
#     all_count.append(dic_count[nam[i]])
# all_count=np.array(all_count)
# # print(all_count)
# all_count=all_count.reshape(len(nam),1)
# # print(all_count)
# kmeans=KMeans(n_clusters=4)
# cls=kmeans.fit_predict(all_count)
# # print(cls)
# # print(metrics.calinski_harabasz_score(all_count,cls))
# centers=kmeans.cluster_centers_
# count_dic={}
# for i in range(len(nam)):
#     count_dic.update({all_count[i,0]:cls[i]})
# col_dic={}
# for i in range(len(nam)):
#     col_dic.update({nam[i]:count_dic[dic_count[nam[i]]]})
#     print(nam[i],dic_count[nam[i]],count_dic[dic_count[nam[i]]])
# print(col_dic)
# print(centers)
# print(all_count)
##########################

max_data,_=torch.max(data,dim=1)
min_data,_=torch.min(data,dim=1)
for i in range(data.size(0)):
    if max_data[i]==min_data[i]:
        data[i]=0
    else:
        data[i]=(data[i]-min_data[i])/(max_data[i]-min_data[i])
    if i>=4:
        data[i]=2*abs(data[i]-0.5)
val=[]
# # # print(data.shape)
# print(data[36,0],data[48,0],data[60,0])
# now_data=torch.concat((data[0].unsqueeze(-1),data[1].unsqueeze(-1),data[46].unsqueeze(-1),data[56].unsqueeze(-1),data[36+30].unsqueeze(-1),data[36+30+38].unsqueeze(-1),data[36+30+38+19].unsqueeze(-1)),dim=1)
# print(now_data.shape)
now_data=data.transpose(0,1)
print(now_data.shape)
for i in range(now_data.size(0)):
    now=0
    for j in range(now_data.size(1)):
        if j==3:
            if now_data[i, j]!=0.5:
                now+=1
                now_data[i, j] = 1
            else:
                now_data[i, j] = 0.2
        elif now_data[i,j]>=0.4:
        # if now_data[i,j]>=0.4:
            now+=1
    val.append(now)

# print(now_data[:,2],now_data[:,3])
# dic_val={'Trp53, Rb1':0, 'Trp53, Rb1, 1':1,'Trp53, Rb1, 1':2,'Trp53, Rb1, Pten':3,'Trp53, Nf1, Rb1':4,'Trp53, Nf1, Rb1, 1':5, 'Trp53, Nf1, Rb1, Pten':6, 'Rb1, 2':7, 'Trp53, Rb1, 2':8, 'Rb1, Pten, 2':9, 'Trp53, Rb1, Pten, 2':10, '0':11, 'Nf1, 0':12, 'Pten, 0':13, 'Nf1, Pten, 0':14, '2':15, 'Trp53, 2':16, 'Pten, 2':17,'Trp53, Pten, 2':18,'1':19,'Trp53, 1':20, 'Nf1, 1':21, 'Pten, 1':22,'Rb1, 1':23,'Trp53, Nf1, 1':24,'Nf1, Pten, 1':25,'Nf1, Rb1, 1':26, 'Rb1, Pten, 1':27, 'Trp53, Pten, 1':28,'Trp53, Nf1, Pten, 1':29, 'Nf1, Rb1, Pten, 1':30, 'Nf1':31, 'Pten':32, 'Rb1':33, 'Trp53':34, 'Trp53, Nf1':35, 'Trp53, Pten':36, 'Rb1, Pten':37, 'Nf1, Pten':38, 'Nf1, Rb1':39, 'Nf1, Rb1, Pten':40, 'Trp53, Nf1, Pten':41, 'Cdkn2a/2b':42,'Pi3kcg':43, 'Pten, Cdkn2a/2b':44, 'Nf1, Cdkn2a/2b':45, 'Nf1, Pten, Cdkn2a/2b':46, 'Pten, Pi3kcg':47, 'Nf1, Pi3kcg':48,'Nf1, Cdkn2a/2b, 0':49,'Cdkn2a/2b, 0':50, 'Nf1, Pten, Cdkn2a/2b, 0':51, 'Pten, Cdkn2a/2b, 0':52   }

# final_dic_val={}
# tot=0
# for _,j in enumerate(dic_val):
#     # print(j)
#     if (len(j)>1 and j[-1]=='0' and j[-2]==' ') or (len(j)==1 and j[0]=='0'):
#         final_dic_val.update({j:tot})
#         tot+=1
# for _,j in enumerate(dic_val):
#     if len(j)>1 and j[-1]=='1' and j[-2]==' ' or (len(j)==1 and j[0]=='1'):
#         final_dic_val.update({j:tot})
#         tot+=1
# for _,j in enumerate(dic_val):
#     if len(j)>1 and j[-1]=='2' and j[-2]==' ' or (len(j)==1 and j[0]=='2'):
#         final_dic_val.update({j:tot})
#         tot+=1
# for _,j in enumerate(dic_val):
#     if len(j)>1 and j[-1]=='3' and j[-2]==' ' or (len(j)==1 and j[0]=='3'):
#         final_dic_val.update({j:tot})
#         tot+=1
# for _,j in enumerate(dic_val):
#     if len(j)>1 and ((j[-1]!='0' and j[-1]!='1' and j[-1]!='2' and j[-1]!='3') or j[-2]!=' '):
#         final_dic_val.update({j:tot})
#         tot+=1
# print(final_dic_val)
import math
dic_val = {
    '0': 0,
    'Pten, 0': 1,
    'Cdkn2a/2b, 0': 2,
    'Nf1, 0': 3,
    'Nf1, Pten, 0': 4,
    'Nf1, Cdkn2a/2b, 0': 5,
    'Pten, Cdkn2a/2b, 0': 6,
    'Nf1, Pten, Cdkn2a/2b, 0': 7,
    'Trp53, 1': 8,
    'Trp53, Pten, 1': 9,
    'Trp53, Nf1, 1': 10,
    'Trp53, Nf1, Pten, 1': 11,
    'Trp53, Rb1, 1': 12,
    'Rb1, 1': 13,
    'Trp53, Nf1, Rb1, 1': 14,
    'Nf1, Rb1, 1': 15,
    'Nf1, 1': 16,
    'Pten, 1': 17,
    'Nf1, Pten, 1': 18,
    'Nf1, Rb1, Pten, 1': 19,
    '1': 20,
    'Rb1, Pten, 1': 21,
    'Trp53, Rb1, Pten, 2': 22,
    '2': 23,
    'Trp53, 2': 24,
    'Rb1, Pten, 2': 25,
    'Pten, 2': 26,
    'Trp53, Pten, 2': 27,
    'Rb1, 2': 28,
    'Trp53, Rb1, 2': 29,
    'Cdkn2a/2b': 30,
    'Nf1, Cdkn2a/2b': 31,
    'Trp53, Nf1, Pten': 32,
    'Pten, Cdkn2a/2b': 33,
    'Trp53, Nf1': 34,
    'Nf1, Pten, Cdkn2a/2b': 35,
    'Rb1, Pten': 36,
    'Trp53, Rb1, Pten': 37,
    'Trp53, Nf1, Rb1': 38,
    'Rb1': 39,
    'Trp53': 40,
    'Trp53, Pten': 41,
    'Pten': 42,
    'Trp53, Rb1': 43,
    'Nf1, Rb1': 44,
    'Nf1': 45,
    'Nf1, Rb1, Pten': 46,
    'Nf1, Pten': 47,
}

# dic_val=final_dic_val
# dic_val={'0': 1, 'Nf1, Cdkn2a/2b, 0': 2, 'Cdkn2a/2b, 0': 3, 'Nf1, Pten, Cdkn2a/2b, 0': 4, 'Pten, Cdkn2a/2b, 0': 5, 'Nf1, 0': 6, 'Nf1, Pten, 0': 7, 'Pten, 0': 8, 'ge1': 9, '1': 10, 'Trp53, Rb1, 1': 11, 'Trp53, Nf1, Rb1, 1': 12, 'Trp53, Nf1, Pten, 1': 13, 'Trp53, Pten, 1': 14, 'Trp53, Nf1, 1': 15, 'Trp53, 1': 16, 'Nf1, 1': 17, 'Nf1, Rb1, 1': 18, 'Rb1, 1': 19, 'Nf1, Pten, 1': 20, 'Pten, 1': 21, 'Nf1, Rb1, Pten, 1': 22, 'Rb1, Pten, 1': 23, 'ge2': 24, '2': 25, 'Trp53, 2': 26, 'Trp53, Pten, 2': 27, 'Pten, 2': 28, 'Trp53, Rb1, Pten, 2': 29, 'Rb1, Pten, 2': 30, 'Trp53, Rb1, 2': 31, 'Rb1, 2': 32, 'ge3': 33, 'Trp53, Rb1': 34, 'Trp53, Pten': 35, 'Trp53, Rb1, Pten': 36, 'Pten': 37, 'Trp53, Nf1, Pten': 38, 'Trp53, Nf1': 39, 'Nf1': 40, 'Trp53': 41, 'Rb1': 42, 'Rb1, Pten': 43, 'Nf1, Pten': 44, 'Cdkn2a/2b': 45, 'Pten, Cdkn2a/2b': 46, 'Nf1, Cdkn2a/2b': 47, 'Trp53, Nf1, Rb1': 48, 'Nf1, Rb1': 49, 'Nf1, Rb1, Pten': 50, 'Nf1, Pten, Cdkn2a/2b': 51}
# dic_num={'Rb1, Pten, 3': 3, 'Pten, 3': 4, 'Rb1, 3': 3, '3': 5, 'Rb1, Pten': 34, 'Pten': 81, 'Rb1': 53, 'Trp53, Rb1, Pten, 2': 9, 'Rb1, Pten, 2': 9, 'Trp53, Pten, 2': 18, 'Pten, 2': 18, 'Trp53, Rb1, 2': 10, 'Rb1, 2': 10, 'Trp53, 2': 19, '2': 19, 'Trp53, Rb1, Pten': 16, 'Trp53, Pten': 46, 'Trp53, Rb1': 34, 'Trp53': 84, 'Trp53, Nf1, Rb1, 1': 17, 'Nf1, Rb1, 1': 33, 'Trp53, Rb1, 1': 17, 'Rb1, 1': 33, 'Trp53, Nf1, 1': 58, 'Nf1, 1': 75, 'Trp53, 1': 58, '1': 75, 'Trp53, Nf1, Rb1': 24, 'Nf1, Rb1': 40, 'Trp53, Nf1': 65, 'Nf1': 98, 'Nf1, Rb1, Pten, 1': 15, 'Rb1, Pten, 1': 15, 'Nf1, Pten, 1': 37, 'Pten, 1': 37, 'Nf1, Rb1, Pten': 22, 'Nf1, Pten': 59, 'Trp53, Nf1, Pten, 1': 21, 'Trp53, Pten, 1': 21, 'Trp53, Nf1, Pten': 28, 'Trp53, Pten, Cdkn2a/2b, 2': 4, 'Pten, Cdkn2a/2b, 2': 4, 'Trp53, Cdkn2a/2b, 2': 4, 'Cdkn2a/2b, 2': 4, 'Trp53, Pten, Cdkn2a/2b': 4, 'Pten, Cdkn2a/2b': 14, 'Trp53, Cdkn2a/2b': 4, 'Cdkn2a/2b': 16, 'Nf1, Pten, Cdkn2a/2b, 0': 9, 'Pten, Cdkn2a/2b, 0': 9, 'Nf1, Cdkn2a/2b, 0': 10, 'Cdkn2a/2b, 0': 10, 'Nf1, Pten, 0': 15, 'Pten, 0': 15, 'Nf1, 0': 16, '0': 16, 'Nf1, Pten, Cdkn2a/2b': 9, 'Nf1, Cdkn2a/2b': 10, 'Trp53, Pten, Pi3kcg, 2': 2, 'Pten, Pi3kcg, 2': 2, 'Trp53, Pi3kcg, 2': 2, 'Pi3kcg, 2': 2, 'Trp53, Pten, Pi3kcg': 2, 'Pten, Pi3kcg': 5, 'Trp53, Pi3kcg': 5, 'Pi3kcg': 8, 'Nf1, Pten, Pi3kcg, 0': 3, 'Pten, Pi3kcg, 0': 3, 'Nf1, Pi3kcg, 0': 3, 'Pi3kcg, 0': 3, 'Nf1, Pten, Pi3kcg': 3, 'Nf1, Pi3kcg': 6, 'Trp53, Nf1, Pi3kcg, 1': 3, 'Nf1, Pi3kcg, 1': 3, 'Trp53, Pi3kcg, 1': 3, 'Pi3kcg, 1': 3, 'Trp53, Nf1, Pi3kcg': 3, 'Nf1, Pten, EgfrvIII, 0': 3, 'Pten, EgfrvIII, 0': 3, 'Nf1, EgfrvIII, 0': 3, 'EgfrvIII, 0': 3, 'Nf1, Pten, EgfrvIII': 3, 'Pten, EgfrvIII': 3, 'Nf1, EgfrvIII': 3, 'EgfrvIII': 3, 'Trp53, Nf1, Rb1, Pten': 7, 'Pten, Cdkn2a/2b, 3': 1, 'Cdkn2a/2b, 3': 1}
# tp=[]
# for i,j in enumerate(dic_val):
#     print(i,j)
#     if j in dic_num:
#         tp.append((j,dic_num[j]))
# print(len(tp))
# print(tp)
for i in range(now_data.size(0)):
    for j in range(i+1,now_data.size(0)):
        # print(i,j)
        # if val[j]==80:
            # print(data[j])
        if dic_val[nam[i]]>dic_val[nam[j]]:
        # if val[i]<val[j]:
            # print(val[i],val[j])
            XX=copy.deepcopy(nam[i])
            nam[i]=copy.deepcopy(nam[j])
            nam[j]=copy.deepcopy(XX)
            YY=copy.deepcopy(now_data[i])
            now_data[i]=copy.deepcopy(now_data[j])
            now_data[j]=copy.deepcopy(YY)
            ZZ=copy.deepcopy(val[i])
            val[i]=copy.deepcopy(val[j])
            val[j]=copy.deepcopy(ZZ)
#             # print(val[i],val[j])
# # print(now_data[2,0],now_data[3,0])
import re

rows_data = []

# # # # === Step 1: Calculate and store feature_count for all rows ===
for i in range(now_data.size(0)):
    tmp = now_data[i].tolist()
    name = nam[i]

#     # === Calculate feature_count ===
    feature_count = 0
    if tmp[0] >= 0.4 and tmp[0] != 100:  # zlz-1
        feature_count += 1
    if tmp[1] >= 0.4 and tmp[1] != 100:  # zlz-2
        feature_count += 1
    if tmp[2] >= 0.4 and tmp[2] != 100:  # qjj-1
        feature_count += 1
    if abs(tmp[3] - 0.2)>0.1:  # qjj-2
        feature_count += 1
    if tmp[4] >= 0.4 and tmp[4] != 100:  # fspl
        feature_count += 1
    if tmp[5] >= 0.4 and tmp[5] != 100:  # sczz
        feature_count += 1

    # Extract the trailing number from name, e.g. name="sample_12" -> group_id=12.
    match = re.search(r'(?:,\s*(\d+)$|^(\d+)$)', name)
    if match:
        group_id = int(match.group(1) or match.group(2))
    else:
        group_id = -1
#     print(group_id, feature_count, name)

    rows_data.append((group_id, feature_count, name, tmp))
# === Step 2: Group by group_id, then sort by feature_count in descending order ===
from collections import defaultdict
grouped = defaultdict(list)

for row in rows_data:
    group_id, feature_count, name, tmp = row
    grouped[group_id].append(row)

# # # Sort within each group
for g in grouped:
    grouped[g].sort(key=lambda x: x[1], reverse=True)
sht.cell(row=1,column=1).value='sample_id'
sht.cell(row=1,column=2).value='Cluster Group'
sht.cell(row=1,column=3).value='Subtype Verhaak'
# sht.cell(row=1,column=4).value='1'
sht.cell(row=1,column=4).value='Microenvironment'
sht.cell(row=1,column=5).value='T-cell rate'
# sht.cell(row=1,column=7).value='2'
sht.cell(row=1,column=6).value='Tumor incidence'
sht.cell(row=1,column=7).value='Median Survival'
sht.cell(row=1,column=8).value='Sig.Fet.Count'

selected_names = set()   # Use a set to remove duplicates.
for i in range(now_data.size(0)):
    tmp=now_data[i].tolist()
    # print(tmp)
    # sht.cell(row=i+2,column=1).value=nam[i]
    feature_count = 0
        # zlz-1 (col=2), zlz-2 (col=3), qjj-1 (col=4), qjj-2 (col=5), fspl (col=6), sczz (col=7)
    if tmp[0] >= 0.4 and tmp[0]!=100:  # zlz-1
        feature_count += 1
    if tmp[1] >= 0.4 and tmp[1]!=100:  # zlz-2
        feature_count += 1
    if tmp[2] >= 0.4 and tmp[2]!=100:  # qjj-1
        feature_count += 1
    if abs(tmp[3] - 0.2)>0.1:  # qjj-2
        feature_count += 1
    if tmp[4] >= 0.4 and tmp[4]!=100:  # fspl
        feature_count += 1
    if tmp[5] >= 0.4 and tmp[5]!=100:  # sczz
        feature_count += 1
    print(tmp)
    print(feature_count)
    # sht.cell(row=i+2, column=8).value = feature_count
    if feature_count >= 3:
        selected_names.add(nam[i])

# Sort within each group
for g in grouped:
    grouped[g].sort(key=lambda x: x[1], reverse=True)

# === Step 3: Write the sorted results to Excel ===
row_idx = 2
for g in [0, 1, 2, -1]:
    print(g)
    if g==-1:
        for _ in range(6):
            sht.cell(row=row_idx, column=1).value=f"ge{_}"
            for ___ in range(7):
                sht.cell(row=row_idx,column=___+2).value=0
            row_idx+=1
    for _, feature_count, name, tmp in grouped[g]:
        sht.cell(row=row_idx, column=1).value = name
        for j in range(len(tmp)):
            if tmp[j] != 100:
                sht.cell(row=row_idx, column=j+2).value = tmp[j]
        sht.cell(row=row_idx, column=8).value = feature_count
        # sht.cell(row=row_idx, column=8).value = (feature_count+1)/6.0
        row_idx += 1

# Save Excel
wb.save(RESULTS_DIR / 'processed_results.xlsx')

# for i in range(now_data.size(0)):
#     tmp=now_data[i].tolist()
#     sht.cell(row=i+2,column=1).value=nam[i]
#     for j in range(now_data.size(1)):
#         if tmp[j]!=100:
#             # if j<=1:
#             sht.cell(row=i+2,column=j+2).value=tmp[j]
#         # === Feature count ===
#     feature_count = 0
#         # zlz-1 (col=2), zlz-2 (col=3), qjj-1 (col=4), qjj-2 (col=5), fspl (col=6), sczz (col=7)
#     if tmp[0] >= 0.4 and tmp[0]!=100:  # zlz-1
#         feature_count += 1
#     if tmp[1] >= 0.4 and tmp[1]!=100:  # zlz-2
#         feature_count += 1
#     if tmp[2] >= 0.4 and tmp[2]!=100:  # qjj-1
#         feature_count += 1
#     if abs(tmp[3] - 0.2)>0.1:  # qjj-2
#         feature_count += 1
#     if tmp[4] >= 0.4 and tmp[4]!=100:  # fspl
#         feature_count += 1
#     if tmp[5] >= 0.4 and tmp[5]!=100:  # sczz
#         feature_count += 1
#     print(feature_count)
        
#     sht.cell(row=i+2, column=8).value = feature_count
# print(nam)
# print(val)
# wb.save('results/processed_results_tmp.xlsx')
