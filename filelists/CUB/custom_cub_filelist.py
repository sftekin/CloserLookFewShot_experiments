import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd()
data_path = join(cwd, 'CUB_200_2011/images')
savedir = './'
dataset_list = ['base', 'val', 'novel']

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
class2file = {}
for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    class2file[folder] = [join(folder_path, cf) for cf in listdir(folder_path)
                          if (isfile(join(folder_path,cf)) and cf[0] != '.')]


with open("split.json", "rb") as f:
    split_dict = json.load(f)

for dataset in dataset_list:
    file_list = []
    label_list = []
    for cls_name in split_dict[dataset]:
        label_idx = int(cls_name.split(".")[0])
        file_list += class2file[cls_name]
        label_list += np.repeat(label_idx, len(class2file[cls_name])).tolist()

    save_dict = {
        "label_names": folder_list,
        "image_names": file_list,
        "image_labels": label_list
    }

    with open(f"{dataset}.json", "w") as f:
        json.dump(save_dict, f)

