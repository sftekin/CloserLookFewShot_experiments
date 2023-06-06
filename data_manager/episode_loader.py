import os
import torch
import json
import random
import numpy as np
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data_manager.transform_loader import TransformLoader


class EpisodeSet(Dataset):
    def __init__(self, meta_file_path, transform, batch_size, load_sampler_indexes=False, max_batch_count=1000):
        self.meta_file_path = meta_file_path
        self.transform = transform
        self.batch_size = batch_size
        self.load_sampler_indexes = load_sampler_indexes
        self.max_batch_count = max_batch_count
        self.dataset = os.path.basename(meta_file_path).split(".")[0]
        self.working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sampler_dir = os.path.join(self.working_dir, "checkpoints", "miniImagenet", "sampler")
        self.sampler_path = os.path.join(self.sampler_dir, f"{self.dataset}_im_sampler.pkl")

        # load the meta file path
        with open(meta_file_path, 'r') as f:
            self.meta = json.load(f)

        # parse the meta file
        self.class2files = {}
        self.labels = set(self.meta['image_labels'])
        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            if y not in self.class2files.keys():
                self.class2files[y] = [x]
            else:
                self.class2files[y].append(x)

        # load sampler if load_sampler_indexes is true else create empty list
        if self.load_sampler_indexes:
            self.sampler_per_class = self.__get_im_sampler()
        else:
            self.sampler_per_class = {}
        create_sampler = False if self.sampler_per_class else True

        self.class_iterator = []
        for label in self.labels:
            filenames = self.class2files[label]
            if create_sampler:
                im_sampler = RandomSampler(n_samples=len(filenames),
                                           batch_size=batch_size,
                                           max_batch_count=max_batch_count)
                self.sampler_per_class[label] = list(im_sampler)
            cls_dataset = ClassDataset(filenames=filenames, label=label, transform=transform)
            data_loader = DataLoader(dataset=cls_dataset,
                                     batch_sampler=self.sampler_per_class[label],
                                     num_workers=0,
                                     pin_memory=False)
            self.class_iterator.append(iter(data_loader))

        # saving the sampler dict
        if not os.path.exists(self.sampler_dir):
            os.makedirs(self.sampler_dir)
        with open(self.sampler_path, "wb") as f:
            pkl.dump(self.sampler_per_class, f)

    def __get_im_sampler(self):
        if not os.path.exists(self.sampler_path):
            print(f"Sampler per image class is NOT found at {self.sampler_path}")
            sampler_dict = {}
        else:
            print(f"Sampler per image class is loaded from {self.sampler_path}")
            with open(self.sampler_path, "rb") as f:
                sampler_dict = pkl.load(f)
        return sampler_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # loads batch size images from the loader of the class i
        item = next(self.class_iterator[i])
        return item


class ClassDataset(Dataset):
    def __init__(self, filenames, label, transform):
        self.filenames = filenames
        self.transform = transform
        self.label = label

    def __getitem__(self, item):
        img = Image.open(self.filenames[item]).convert('RGB')
        img = self.transform(img)
        return img, self.label

    def __len__(self):
        return len(self.filenames)


class RandomSampler:
    def __init__(self, n_samples, batch_size, max_batch_count):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.max_batch_count = max_batch_count

    def __len__(self):
        return self.max_batch_count

    def __iter__(self):
        for i in range(self.max_batch_count):
            yield torch.randperm(self.n_samples)[:self.batch_size]


class EpisodeBatchSampler:
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


def get_episode_loader(meta_file_path, image_size, n_episodes, augmentation,
                       n_way, n_shot, n_query, num_workers, load_sampler_indexes=False):
    # create the transformer and the episode creator
    transform = TransformLoader(image_size=image_size,
                                augmentation=augmentation)
    dataset = EpisodeSet(meta_file_path=meta_file_path,
                         transform=transform.get_transform(),
                         batch_size=n_shot + n_query,
                         max_batch_count=n_episodes,
                         load_sampler_indexes=load_sampler_indexes)

    # load the batch sampler if load_sampler_indexes is true
    batch_sampler_path = os.path.join(dataset.sampler_dir, f"{dataset.dataset}_{n_episodes}_batch_sampler.pkl")
    if load_sampler_indexes and os.path.exists(batch_sampler_path):
        with open(batch_sampler_path, "rb") as f:
            batch_sampler = pkl.load(f)
    else:
        if load_sampler_indexes:
            print(f"batch sampler is not found at the path {batch_sampler_path},"
                  f" new sampler is created")
        batch_sampler = EpisodeBatchSampler(n_classes=len(dataset),
                                            n_way=n_way,
                                            n_episodes=n_episodes)
        batch_sampler = list(batch_sampler)
        with open(batch_sampler_path, "wb") as f:
            pkl.dump(batch_sampler, f)

    # lastly create the episode data loader.
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             pin_memory=False)
    return data_loader


def get_sample_indexes(dataset, batch_sampler):
    my_batch = []
    track = {c: 0 for c in dataset.labels}
    for batch_classes in batch_sampler:
        batch = []
        for i in batch_classes:
            label = list(dataset.labels)[i]
            idx = track[label]
            samples = dataset.sampler_per_class[label][idx]
            track[label] += 1
            batch.append(samples)
        my_batch.append(torch.stack(batch))
    return my_batch


if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    way = 5
    shot = 1
    query = 15
    ep = 100
    num_workers = 1
    im_size = 84
    meta_file = os.path.join(cur_dir, "filelists", "miniImagenet", "base.json")

    trs = TransformLoader(image_size=im_size)
    ds = EpisodeSet(meta_file_path=meta_file, transform=trs.get_transform(),
                    batch_size=shot + query, load_sampler_indexes=False)
    smp = EpisodeBatchSampler(n_classes=len(ds), n_way=way, n_episodes=ep)

    batch_sampler_path = os.path.join(ds.sampler_dir, "batch_sampler.pkl")
    rand_samples = list(smp)
    with open(batch_sampler_path, "wb") as f:
        pkl.dump(rand_samples, f)
    dl = DataLoader(dataset=ds, batch_sampler=rand_samples, num_workers=num_workers, pin_memory=True)

    assert_x_sum = 0
    assert_y_sum = 0
    for j, (x, y) in enumerate(dl):
        print(f"{j}/{len(dl)}")
        assert x.shape == torch.Size([way, shot + query, 3, im_size, im_size])
        assert y.shape == torch.Size([way, shot + query])
        assert_y_sum += y.sum()
        assert_x_sum += x.sum()
        if j == 0:
            break

    ds = EpisodeSet(meta_file_path=meta_file, transform=trs.get_transform(),
                    batch_size=shot + query, load_sampler_indexes=True)
    with open(batch_sampler_path, "rb") as f:
        rand_samples = pkl.load(f)
    dl = DataLoader(dataset=ds, batch_sampler=rand_samples, num_workers=num_workers, pin_memory=True)

    print(x[-1, -1, 0, 0, 0], y)
    x, y = next(iter(dl))
    print(x[-1, -1, 0, 0, 0], y)

    assert x.sum() == assert_x_sum
    assert y.sum() == assert_y_sum

