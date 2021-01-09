import cv2
import glob
import os
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import pickle

def process_jpgs(csv_path,video_path,out_path_root,class_idx_path):
    with open(class_idx_path, 'rb') as f_in:
        class_idx = pickle.load(f_in)

    df = pd.read_csv(csv_path)
    df = df.sort_values('label')
    df.drop(['Unnamed: 0','index'],axis=1,inplace=True)
    for (j,x) in tqdm(df.iterrows(),total=len(df)):
        start_frame = x['start_frame']
        end_frame = x['end_frame']
        if start_frame != end_frame:
            if start_frame == 0: start_frame+=1
            outpath = os.path.join(out_path_root,'v_{}/{}'.format(x['video_id'],class_idx[x['label']]))
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            for i in range(start_frame,end_frame):
                outname = os.path.join(outpath, 'frame_{}.png'.format(i))
                # if not os.path.exists(outname):
                frame_name = os.path.join(video_path, 'v_{}'.format(x['video_id']),
                                          'image_{}.jpg'.format(str(i).zfill(5)))
                img = cv2.imread(frame_name, 0)
                if img is None: continue
                img = cv2.resize(img, (224, 224))
                cv2.imwrite(outname, img)
                # else:
                #     continue




def flatten(d, sep="_"):
    import collections

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            lens = len(t)
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)

    return obj

def make_csv(database,video_path):
    #
    dict_base = {}
    i = 0
    for video_id in tqdm(database):
        temp_dict = database[video_id]

        path = os.path.join(video_path, "v_%s" % video_id)
        frames_names = glob.glob(os.path.join(path, '*.jpg'))
        nr_frames = len(frames_names)
        fps = (nr_frames * 1.0) / temp_dict['duration']

        for j in range(len(temp_dict['annotations'])):
            temp = {}
            temp['video_id'] = video_id
            temp['fps'] = fps
            for key in temp_dict:
                if key != 'annotations':
                    temp[key] = temp_dict[key]
                else:
                    temp['segment_start'] = temp_dict[key][j]['segment'][0]
                    temp['segment_end'] = temp_dict[key][j]['segment'][1]
                    temp['label'] = temp_dict[key][j]['label']

                    temp['start_frame'], temp['end_frame'] = int(temp['segment_start'] * fps), int(temp['segment_end'] * fps)
            dict_base[i] = temp
            i += 1

    df = pd.DataFrame.from_dict(dict_base,orient='index')
    df.reset_index(level=0, inplace=True)
    df.to_csv('./activity_net.csv')


def make_class_idx(data):
    classes_idx = {}
    taxonomy = data["taxonomy"]
    for entry in taxonomy:
        classes_idx[entry['nodeName']] = entry['nodeId']
    with open('./class_indx.pkl', "wb") as fin:
        pickle.dump(classes_idx, fin, protocol=4)


if __name__=='__main__':
    json_path = "../hyperbolic_action-master/activity_net.v1-3.json"
    json_path = json_path
    with open(json_path, "r") as fobj:
        data = json.load(fobj)
    database = data["database"]

    video_path = "/data/Activity_net/Video_jpg"
    out_path_root = "/data/Activity_net/processed_jpg_224"
    class_idx_path = './class_indx.pkl'
    csv_path = './activity_net.csv'
    process_jpgs(csv_path,video_path,out_path_root,class_idx_path)
    # make_csv(database,video_path)
    # make_class_idx(data)

