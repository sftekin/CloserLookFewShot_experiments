import os
import glob
import json
import numpy as np

cur_dir = os.getcwd()
with open("split.json", "rb") as f:
    split_dict = json.load(f)
all_labels = []
for class_code in split_dict.values():
    all_labels += class_code
all_labels = sorted(all_labels)

dataset_list = ['base', 'val', 'novel']
for dataset in dataset_list:
    file_list = []
    label_list = []
    for class_code in split_dict[dataset]:
        cls_id, cls_name = class_code.split('.')
        file_names = [os.path.join(cur_dir, fn) for fn in glob.glob(os.path.join("images", f"{cls_name}*.jpg"))]
        label_list += np.repeat(int(cls_id), len(file_names)).tolist()
        file_list += file_names

    save_dict = {
        "label_names": all_labels,
        "image_names": file_list,
        "image_labels": label_list
    }

    with open(f"{dataset}.json", "w") as f:
        json.dump(save_dict, f)


print()

