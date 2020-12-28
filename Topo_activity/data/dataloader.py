import torch
import pickle
import pandas as pd
import numpy as np
import json
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict as ddict
from graph_dataset import BatchedDataset


class TreeDataset(Dataset):
    def __init__(self,json_path,**kwargs):
        super(TreeDataset, self).__init__()
        self.json_path = json_path
        with open(self.json_path, "r") as fobj:
            self.data = json.load(fobj)
        self.database = self.data["database"]
        self.taxonomy = self.data["taxonomy"]

        # Dataset Hparams
        self.negs = kwargs.get('negs',50)
        self.batch_size = kwargs.get('batch_size',12800)
        self.ndproc = kwargs.get('ndproc',8) # 'Number of data loading processes'
        self.burnin = kwargs.get('burnin',20)
        self.dampening = kwargs.get('dampening',0.75) # 'Sample dampening during burnin'
        self.neg_multiplier = kwargs.get('neg_multiplier',1.0)


        self.idx, self.objects, self.weights = self.get_hypernymity()
        # self.videos = self.get_videos()

    def get_graph_dataset(self):

        data = BatchedDataset(self.idx, self.objects, self.weights, self.negs, self.batch_size,
                          self.ndproc, self.burnin > 0, self.dampening)
        data.neg_multiplier = self.neg_multiplier

        return data

    def get_hypernymity(self):
        hypers = []
        for item in self.taxonomy:
            hypers.append((item['nodeName'], item['parentName']))
        df = pd.DataFrame(hypers,columns=['child','parent'])
        idx, objects = pd.factorize(df[['child', 'parent']].values.reshape(-1),na_sentinel=None)
        idx = idx.reshape(-1, 2).astype('int')
        weights = np.ones([len(idx)])
        return idx, objects.tolist(), weights

    def get_videos(self,activity,subset):
        videos = []
        for x in self.database:
            if self.database[x]["subset"] != subset: continue
            xx = random.choice(self.database[x]["annotations"])
            if xx["label"] == activity:
                yy = {"videoid": x, "duration": self.database[x]["duration"],
                      "start_time": xx["segment"][0], "end_time": xx["segment"][1]}
                videos.append(yy)
        return random.choice(videos)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--json_path', type=str, default="../hyperbolic_action-master/activity_net.v1-3.json")
    parser.add_argument('--video_path', type=str, default="/data/Activity_net/Video_jpg")

    args = parser.parse_args()
    dataset = TreeDataset(args.json_path)
    data = dataset.get_graph_dataset()

