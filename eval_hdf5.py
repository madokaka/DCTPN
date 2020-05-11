#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:33:57 2018

@author: madoka
"""
import os
import torch
import h5py
import numpy as np
import progressbar
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import json
from collections import OrderedDict
# torch.cuda.set_device(1)
from TCN_test import TCN
import time
num_chans = [2048,1024,512,256,128,64]
model = TCN(4096,num_chans)

device_ids = [0,1,2,3]

model = torch.nn.DataParallel(model, device_ids=device_ids)
model.cuda()


model.eval()
features = h5py.File('output_frm/c3d_features_train_1_test.hdf5')

video_list = []
for k in features.keys():
    video_list.append(features[k].name.replace('/',''))

class Datasplit(Dataset):
    
    def collate_fn(self,data):
        
        features = [d[0] for d in data]

        
        return torch.cat(features,0)
    
    
    def __init__(self,video_ids,features):
        self.video_ids = video_ids
        self.features = features

        
    def __getitem__(self,index):
        video_id = self. video_ids[index]
        features = self.features[video_id]['c3d_features']
        nfeats = features.shape[0]
        nwindows = max(1,nfeats - 128 + 1)
        sample = range(nwindows)
        
        feature_windows = np.zeros((nwindows,128,features.shape[1]))

        for j,w_start in enumerate(sample):
            w_end = min(w_start+128,nfeats)
            feature_windows[j,0:w_end-w_start,:] = features[w_start:w_end,:]

        return torch.FloatTensor(feature_windows)

    def __len__(self):
        
        return len(self.video_ids)
    

def iou(interval, featstamps, return_index=False):
    start_i, end_i = interval[0], interval[1]
    output = 0.0
    gt_index = -1
    for i, (start, end) in enumerate(featstamps):
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
        overlap = float(intersection) / (union + 1e-8)
        if overlap >= output:
            output = overlap
            gt_index = i
    if return_index:
        return output, gt_index
    return output

def score(inputs_list):
    score = []
    for i in inputs_list:
        features_inputs = i.cuda()
        features_inputs = Variable(features_inputs)
        outputs = model(features_inputs)
        outputs = outputs.cpu()
        batch_size,nb_steps,k = outputs.size()
        sort,_ = outputs.view(batch_size*nb_steps*k).sort()
        sort = sort.detach().numpy()
        score.append(sort[-10])

    return score
def split_batch(features_windows):
    inputs_list= []
    times_list = []
    batch_size,num_step,k = features_windows.size()
    if batch_size%100 == 0:
        num_r = batch_size/100
        num_r = int(num_r)
    else:
        num_r = batch_size//100 +1
    for bat_r in range(num_r):
        starts = max(0,100*bat_r)
        ends = min(batch_size,100*(bat_r+1))
        inputs = features_windows[starts:ends]
        inputs_list.append(inputs)
        times_list.append((starts,ends))
    return inputs_list,times_list



def outputs_times(inputs_list):
    score_t = score(inputs_list)
    score_t.sort()
    if len(score_t) < 10:
        num_e = len(score_t)
    else:
        num_e = 10
    score_num = torch.from_numpy(np.array(score_t[-num_e])).cuda()

    timestamps = []
    
    for num,i in enumerate(inputs_list):

        features_inputs = i.cuda()
        features_inputs = Variable(features_inputs)
        outputs = model(features_inputs)
        fgroup = mieba.create_group(video_list[num])


        batch,nb_step,k = outputs.shape
        outputs = outputs> score_num
        for batch_s in range(batch):
            for time_step in range(nb_step):
                p = outputs[batch_s,time_step]
                if p.sum() != 0:
                    end = time_step+batch_s+times[num][0]
                    for k_s in range(64):
                        if p[k_s] != 0:
                            start = time_step-k_s-1+batch_s+times[num][0]
                            timestamps.append((start,end))
     
    outputs_timestamps = set(timestamps)
    return outputs_timestamps

def json_read():
    time_list = []
    time_dict = {}
    json_annotations = json.load(open('activity_net.v1-3.min.json'))

    for v_n in json_annotations['database'].keys():
        annotations = json_annotations['database'][v_n]['annotations']
        timestamps = [ann['segment'] for ann in annotations]
        time_dict[v_n] = timestamps
        time_list.append(timestamps)
    return time_list


def nms(outputs_list,overlap_num=0.8):
    nms_list = []
    for i in outputs_list[0]:
        nms_list.append(i)
        for a in nms_list:
            start = i[0]
            end = i[1]
            start_a = a[0]
            end_a = a[1]
            intersection  = max(0,min(end,end_a)-max(start,start_a))
            union = min(max(end,end_a)-min(start,start_a),end-start+end_a-start_a)
            overlap = float(intersection)/(union+1e-8)
            if overlap>overlap_num:
                if i==a:
                    pass
                else:
                    nms_list.remove(a)
    return nms_list
                    



mieba = h5py.File('6layer_128_validation.hdf5','w')

evaluate_dataset = Datasplit(video_list,features)


outputs_times_list = []


start_time = time.time()
recall = np.zeros(len(evaluate_dataset))
for num_batch,features_windows in enumerate(evaluate_dataset):
    inputs_list,times = split_batch(features_windows)
    outputs_times_list.append(outputs_times(inputs_list))
    gt_detected = np.zeros(len(json_read()[num_batch]))
    for i,timee in enumerate(outputs_times_list[num_batch]):
        iou_i,k = iou(timee,json_read()[num_batch],return_index=True)
        if iou_i>0.5:
            gt_detected[k] = 1
    recall[num_batch] = gt_detected.sum()*1./len(gt_detected)    

