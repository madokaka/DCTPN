# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from TCN_test import TCN
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import progressbar
parser = argparse.ArgumentParser()

parser.add_argument('--dataset',type=str,default='ActivityNet')
parser.add_argument('--data',type=str,default='activity_net.v1-3.min.json')
parser.add_argument('--features',type=str,default='c3d_features_Activitynet_training.hdf5')
parser.add_argument('--labels',type=str,default='acitivity_labels_128.hdf5')
parser.add_argument('--vid_ids',type=str,default='activity_video_ids_128.json')
parser.add_argument('--k',type=int,default=64)
args = parser.parse_args()



train_loaded = True

torch.manual_seed(1111)
device_ids = [0, 1, 2, 3]
num_chans = [2048,1024,512,256,128,64]


class ProposalDataset(object):

    def __init__(self,args):

        self.inter = 0.5
        self.data = json.load(open(args.data))
        self.features = h5py.File(args.features)
        if not os.path.exists(args.labels) or not os.path.exists(args.vid_ids):
            self.generate_labels(args)
            print('generate label')
        print('label exist')
        self.labels = h5py.File(args.labels)
        self.vid_ids = json.load(open(args.vid_ids))
        self.durations = {}
        self.gt_times ={}
        self.w1 = self.vid_ids['w1']
        for split in ['training','validation','testing']:
            setattr(self,split+'_ids',self.vid_ids[split])
            for video_id in self.vid_ids[split]:
                self.durations[video_id] = self.data['database'][video_id]['duration']
                self.gt_times[video_id] = [ann['segment'] for ann in self.data['database'][video_id]['annotations']]
                
        
            
        
    def generate_labels(self,args):
        
        label_dataset = h5py.File(args.labels,'w')
        bar = progressbar.ProgressBar(maxval=len(self.data['database'].keys())).start()
        prop_captured = []
        prop_pos_examples = []
        video_ids = self.data['database'].keys()
        split_ids = {'training': [], 'validation': [], 'testing': [],
                     'w1': []}
        for progress,video_id in enumerate(video_ids):
            try:
                features = self.features['v_' + video_id]['c3d_features']
                nfeats = features.shape[0]
                duration = self.data['database'][video_id]['duration']
                annotations = self.data['database'][video_id]['annotations']
                timestamps = [ann['segment'] for ann in annotations]
                featstamps = [self.timestamp_to_featstamp(x,nfeats,duration) for x in timestamps]
                nb_prop = len(featstamps)
                # 从尾部删除
                for i in range(nb_prop):
                    if (featstamps[nb_prop-i-1][1] - featstamps[nb_prop-i-1][0]>args.k/0.5):
                        del featstamps[nb_prop-i-1]
                if len(featstamps) == 0:
                    if len(timestamps) == 0:
                        prop_captured +=[-1.0]
                    else:
                        prop_captured +=[0.0]
                    continue
                split_ids[self.data['database'][video_id]['subset']] += [video_id]
                labels = np.zeros((nfeats,args.k))
                gt_captured = []
                for t in range(nfeats):
                    for k in range(args.k):
                        iou,gt_index = self.iou([t-(k/self.inter),t+1],featstamps,return_index=True)
                        if iou >= 0.5:
                            labels[t,k] = 1

                            gt_captured += [gt_index]
                prop_captured += [1.* len(np.unique(gt_captured))/len(timestamps)]
                if self.data['database'][video_id]['subset'] == 'training':
                    prop_pos_examples += [np.sum(labels,axis=0)*1./nfeats]
                video_dataset = label_dataset.create_dataset(video_id,(nfeats,args.k),dtype='f')
                video_dataset[...] = labels
                bar.update(progress)
            except:
                print('pass vidoe_id:',video_id)
                pass
        split_ids['w1'] = np.array(prop_pos_examples).mean(axis=0).tolist()
        json.dump(split_ids,open(args.vid_ids,'w'))
        bar.finish()
                    
            
            
    def timestamp_to_featstamp(self,timestamp,nfeats,duration):
        start,end = timestamp 
        start = min(int(round(start/duration*nfeats)),nfeats - 1)
        end = max(int(round(end/duration*nfeats)),start + 1)
        return start,end
    
    def iou(self,interval,featstamps,return_index=False):

        start_i,end_i = interval[0],interval[1]        
        output = 0.0
        gt_index = -1
        for i,(start,end) in enumerate(featstamps):
            intersection = max(0,min(end,end_i) - max(start,start_i))
            union = min(max(end,end_i) - min(start,start_i),end- start + end_i - start_i)
            overlap = float(intersection)/(union+1e-8)
            if overlap >= output:
                output = overlap
                gt_index = i
        if return_index:
            return output,gt_index
        return output



class Datasplit(Dataset):
    
    def collate_fn(self,data):
        
        features = [d[0] for d in data]
        masks = [d[1] for d in data]
        labels = [d[2] for d in data]
        
        return torch.cat(features,0),torch.cat(masks,0),torch.cat(labels,0)
    
    
    def __init__(self,video_ids,dataset,args):
        self.video_ids = video_ids
        self.features = dataset.features
        self.labels = dataset.labels
        self.durations = dataset.durations
        
    def __getitem__(self,index):
        video_id = self. video_ids[index]
        features = self.features['v_'+video_id]['c3d_features']
        labels = self.labels[video_id]
        nfeats = features.shape[0]
        nwindows = max(1,nfeats - 128 + 1)
        sample = range(nwindows)
        if nwindows > 256:
            sample = np.random.choice(nwindows,256)
            nwindows = 256
            
        mask = np.zeros((256,128,args.k))
        for i in range(128):
            for a in range((i + 3) // 2):
                mask[:, i, :min(64, a)] = 1

        mask = torch.FloatTensor(mask)
        
        
        mask = mask[:nwindows,:,:]
        
        feature_windows = np.zeros((nwindows,128,features.shape[1]))
        label_windows = np.zeros((nwindows,128,args.k))


        for j,w_start in enumerate(sample):
            w_end = min(w_start+128,nfeats)
            feature_windows[j,0:w_end-w_start,:] = features[w_start:w_end,:]
            label_windows[j,0:w_end-w_start,:] = labels[w_start:w_end,:]


            
        return torch.FloatTensor(feature_windows),mask,torch.Tensor(label_windows)

    def __len__(self):
        
        return len(self.video_ids)

class Evaluatesplit(Dataset):

    def __init__(self,video_ids,dataset,args):
        self.video_ids = video_ids
        self.features = dataset.features
        self.labels = dataset.labels
        self.durations = dataset.durations
        self.gt_times = dataset.gt_times
    
    def collate_fn(self,data):
        features = data[0][0]
        gt_times = data[0][1]
        durations = data[0][2]
        
        return features.view(1,features.size(0),features.size(1)),gt_times,durations
    

    def __getitem__(self,index):
        video_id = self.video_ids[index]
        features = self.features['v_'+video_id]['c3d_features']
        duration = self.durations[video_id]
        gt_times = self.gt_times[video_id]
        
        return torch.FloatTensor(features),gt_times,duration
    

    def __len__(self):
        return len(self.video_ids)


def train(epoch,w1):
    model.train()
    total_loss = []
    model.train()
    start_time = time.time()
    for batch_idx,(features,masks,labels) in enumerate(train_loader):
        features = features.cuda()
        labels = labels.cuda()
        masks = masks.cuda()
        features = Variable(features)
        optimizer.zero_grad()
        # print(features.size())
        proposals = model(features)

        loss = compute_loss_with_BCE(proposals,masks,labels,w1)

    
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        if batch_idx % 100 == 0:
            cur_loss = total_loss[-1]
            log_entry = ('epoch{:3d} | {:5d}/{:5d} | lr {:2.4f} | \
                  loss {:5.6f} | time: {:5.2f}').format(epoch,batch_idx,len(train_loader),0.1,cur_loss*1000,
                  (time.time()-start_time))
            print(log_entry)
            with open('train_128.log','a') as f:
                f.write(log_entry)
                f.write('\n')
            start_time = time.time()

def evaluate(data_loader,maximum=None):
    model.eval()
    total = len(data_loader)
    if maximum is not None:
        total = min(total,maximum)
    recall = np.zeros(total)
    for batch_idx,(f,g,d) in enumerate(data_loader):
        if maximum is not None and batch_idx >= maximum:
            break
        f = f.cuda()
        features = Variable(f)
        proposals = model(features)
        recall[batch_idx] = calculate_stats(proposals,g,d,args)
    return np.mean(recall)

def calculate_stats(proposals,gt_times,duration,args):
    timestamps = proposals_to_timestamps(proposals.data,duration)
    gt_detected = np.zeros(len(gt_times))
    for i,timestamp in enumerate(timestamps):
        iou_i,k = dataset.iou(timestamp,gt_times,return_index=True)
        if iou_i >0.5:
            gt_detected[k] = 1
    return gt_detected.sum()*1./len(gt_detected)

def proposals_to_timestamps(proposals,duration,num_proposals=None):
    _,nb_steps,k = proposals.size()
    if num_proposals and num_proposals < nb_steps*k:
        sort,_ = proposals.view(nb_steps*k).sort()
        score_threshold = sort[-num_proposals]
        proposals = proposals >= score_threshold
    step_length = duration/nb_steps
    timestamps = []
    for time_step in np.arange(nb_steps):
        p = proposals[0,time_step]
        if p.sum() != 0:
            end = time_step*step_length
            for k in np.arange(k):
                if p[k] != 0:
                    start = max(0,time_step-k-1)*step_length
                    timestamps.append((start,end))
    return timestamps

def compute_loss_with_BCE(outputs,masks,labels,w1):
        
    w1 = torch.FloatTensor(w1).type_as(outputs.data)
    
    w0 = 1.0 - w1
    labels = labels.mul(masks)
    
    weights = labels.mul(w0.expand(labels.size()))+(1.-labels).mul(w1.expand(labels.size()))
    weights = weights
    masks = Variable(masks)
    labels = Variable(labels)
    outputs = outputs.mul(masks)
    criterion = torch.nn.BCELoss(weight=weights)
    loss = criterion(outputs,labels)
    
    return loss


        
dataset = ProposalDataset(args)
w1 = dataset.w1
train_dataset = Datasplit(dataset.training_ids, dataset, args)
val_dataset = Evaluatesplit(dataset.validation_ids,dataset,args)
train_loader = DataLoader(train_dataset,batch_size=1,collate_fn=train_dataset.collate_fn)


if train_loaded == True:
    model = TCN(4096, num_chans)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    model.load_state_dict(torch.load('save_load_test.pth'))
else:
    model = TCN(4096,num_chans)
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=device_ids)




epoch_num = 0

optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0)
for epoch in range(epoch_num,10000):
    epoch_start_time = time.time()
    train(epoch,w1)
    if epoch % 10 == 0  and epoch >0:
       

        log_entry = ('|end of epoch {:3d} | time:{:5.2f}s | '.format(epoch,(time.time()-epoch_start_time)))
        print('='*89)
        print(log_entry)
        print('='*89)
        with open('val_6layer_train_1.log', 'a') as f:
            f.write(log_entry)
            f.write('\n')

        torch.save(model.state_dict(), os.path.join('6layer_64_2_model_' + str(epoch) + '.pth'))

        print('save_model')
