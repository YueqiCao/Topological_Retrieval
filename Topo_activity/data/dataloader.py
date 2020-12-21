import torch
import pickle
import pandas as pd
import numpy as np
import json
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict as ddict


class Dataset(Dataset):
    def __init__(self,json_path,**kwargs):
        super(TreeDataset, self).__init__()
        self.json_path = json_path
        with open(self.json_path, "r") as fobj:
            self.data = json.load(fobj)
        self.database = self.data["database"]
        self.taxonomy = self.data["taxonomy"]

        self.dx, self.objects, self.weights = self.get_hypernymity()
        self.videos = self.get_videos()

    def get_hypernymity(self):
        hypers = []
        for item in self.taxonomy:
            hypers.append((item['nodeName'], item['parentName']))
        df = pd.DataFrame(hypers,columns=['child','parent'])
        idx, objects = pd.factorize(df[['child', 'parent']].values.reshape(-1))
        idx = idx.reshape(-1, 2).astype('int')
        weights = np.ones([len(idx)])
        return idx, objects.tolist(), weights

    def get_videos(self,activity,subset):
        videos = []
        for x in self.database:
            if self.database[x]["subset"] != subset: continue
            xx = random.choice(database[x]["annotations"])
            if xx["label"] == activity:
                yy = {"videoid": x, "duration": database[x]["duration"],
                      "start_time": xx["segment"][0], "end_time": xx["segment"][1]}
                videos.append(yy)
        return random.choice(videos)

class TreeDataset(Dataset):
    _neg_multiplier = 1
    _ntries = 10
    _sample_dampening = 0.75
    def __init__(self, nnegs, unigram_size=1e8):
        super(TreeDataset, self).__init__()
        assert self.idx.ndim == 2 and self.idx.shape[1] == 2
        assert self.weights.ndim == 1
        assert len(self.idx) == len(self.weights)
        assert nnegs >= 0
        assert unigram_size >= 0
        self.nnegs = nnegs
        self.burnin = False

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(len(self.objects), dtype=np.float)
        self.max_tries = self.nnegs * self._ntries
        for i in range(self.idx.shape[0]):
            t, h = self.idx[i]
            self._counts[h] += self.weights[i]
            self._weights[t][h] += self.weights[i]
        self._weights = dict(self._weights)
        nents = int(np.array(list(self._weights.keys())).max())
        assert len(self.objects) > nents, f'Number of objects do no match'
        if unigram_size > 0:
            c = self._counts ** self._sample_dampening
            self.unigram_table = np.random.choice(
                len(self.objects),
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.shape[0]

    def weights(self, inputs, targets):
        return self.fweights(self, inputs, targets)

    def nnegatives(self):
        if self.burnin:
            return self._neg_multiplier * self.nnegs
        else:
            return self.nnegs

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return th.cat(inputs, 0), th.cat(targets, 0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--json_path', type=str, default="../hyperbolic_action-master/activity_net.v1-3.json")
    parser.add_argument('--video_path', type=str, default="/data/Activity_net/Video_jpg")

    args = parser.parse_args()
    TreeDataset(args.json_path)

