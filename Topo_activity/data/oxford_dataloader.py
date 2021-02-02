import torch
import pickle
import pandas as pd
import numpy as np
import json
import random
from torch.utils.data import Dataset, DataLoader
import torchvision
from collections import defaultdict as ddict
import glob
import os
import imageio
from PIL import Image
from tqdm import tqdm
import cv2
import re

class OxfordBuildings(Dataset):
    def __init__(self, args):
        super(OxfordBuildings, self).__init__()
        self.args = args
        if not os.path.exists(os.path.join(self.args.image_path,'df.csv')):
            self.image_files = glob.glob(os.path.join(args.image_path,'*.jpg'))
            self.gt_files = glob.glob(os.path.join(args.gt_path, '*_query.txt'))
            self.df, self.df_query = self.preprocess_()
        else:
            self.df = pd.read_csv(self.args.image_path+'df.csv')
            # self.df_gt = pd.read_csv(self.args.gt_path+'df_gt.csv')
            self.df_query = pd.read_csv(self.args.gt_path+'df_test.csv')


    def preprocess_(self):
        d = []
        q = []
        temp = []
        for file in self.gt_files:

            # if 'query' not in file:
            #     data = pd.read_csv(file, sep=" ", header=None)
            #
            file_ = os.path.basename(file)
            file_ = file_.split('.')[0]
            file_ = re.match("(\D+)_(\d+)_(\D+)",file_).groups()
            #     if file_:
            #
            #         dic = {'class': file_[0],
            #                 'subclass': file_[1],
            #                'quality':file_[-1],
            #                'for_retrieve' : '{}_{}'.format(file_[0],file_[1]),
            #                'gt': list(data[0])
            #         }
            #         d.append(dic)
            # else:
            data = pd.read_csv(file, sep=" ", header=None)
            for index,row in data.iterrows():
                bbox = [row[i] for i in range(1,5)]
                cls =re.match("(\D+)(\d+)_(\D+)_(\d+)",row[0]).groups()
                gt = []
                class_ = file_[0]
                subclass = file_[1]

                for quality in ['good','ok']:
                    gt_file = os.path.join(self.args.gt_path,'{}_{}_{}.txt'.format(class_,subclass,quality))
                    data = pd.read_csv(gt_file, sep=" ", header=None)
                    gt.append(list(data[0]))
                gt_good = gt[0]
                gt_ok = gt[1]
                dic = {
                    'class':class_,
                    'subclass':subclass,
                    'path': os.path.join(self.args.image_path,'{}_{}.jpg'.format(cls[2],cls[-1])),
                    'bounding_box':bbox,
                    'gt_good':gt_good,
                    'gt_ok':gt_ok,
                    'gt':gt
                }
                temp.append('{}_{}'.format(cls[1], cls[2]))
                q.append(dic)

        df_query = pd.DataFrame(q)
        df_query.to_csv(self.args.gt_path + 'df_test.csv')

        d = []
        for file in self.image_files:
            file_ = os.path.basename(file)
            file_ = file_.split('.')[0]
            if file_ not in  temp:
                file_ = re.match("(\D+)_(\d+)", file_).groups()

                dic = {'class': file_[0],
                       'id': file_[1],
                       'path': file}
                d.append(dic)
        df = pd.DataFrame(d)
        df.to_csv(self.args.image_path+'df.csv')


        return df, df_query



    def __getitem__(self, item):
        raise NotImplementedError

class TestOxfordBuildings(OxfordBuildings):
    def __init__(self, args):
        super(TestOxfordBuildings, self).__init__(args)

    def __len__(self):
        return len(self.df_query)

    def __getitem__(self, item):
        row = self.df_query.iloc[item]
        img = cv2.imread(row['path'], 0)
        img = cv2.resize(img, (64, 64))
        img = img / 255.
        img = torch.from_numpy(img)

        label = row['gt']
        return img, label

class TrainOxfordBuildings(OxfordBuildings):
    def __init__(self, args):
        super(TrainOxfordBuildings, self).__init__(args)
        fin = int(0.8*len(self.df))
        self.train = self.df.iloc[0:fin]
        self.val = self.df.iloc[fin:-1]

    def __len__(self):
        return int(0.8*len(self.df))

    def __getitem__(self, item):
        row = self.train.iloc[item]
        img = imageio.imread(row['path'])[...,0:1]
        img = cv2.resize(img, (64, 64))
        img = img / 255.
        img = img[...,np.newaxis]
        img = np.transpose(img,(2,0,1))
        img = torch.from_numpy(img)
        return img

class ValOxfordBuildings(OxfordBuildings):
    def __init__(self, args):
        super(ValOxfordBuildings, self).__init__(args)
        fin = int(0.8*len(self.df))
        self.val = self.df.iloc[fin:-1]

    def __len__(self):
        return int(0.2*len(self.df))

    def __getitem__(self, item):
        row = self.val.iloc[item]
        img = imageio.imread(row['path'])[...,0:1]
        img = cv2.resize(img, (64, 64))
        img = img / 255.
        img = img[..., np.newaxis]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        return img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--image_path', type=str, default="/vol/medic01/users/av2514/Pycharm_projects/Datasets/Oxford_Buildings/images")
    parser.add_argument('--gt_path', type=str, default="/vol/medic01/users/av2514/Pycharm_projects/Datasets/Oxford_Buildings/ground_truth")


    args = parser.parse_args()
    # dataset = TreeDataset(args.json_path)
    dataset = OxfordBuildings(args)
    dataset.__getitem__(100)
    # data = dataset.get_graph_dataset()
