import numpy as np
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data_manager.transform_loader import TransformLoader


class DatasetIM(Dataset):
    def __init__(self, meta_file_path, transform):
        self.transform = transform

        with open(meta_file_path, 'r') as f:
            self.meta = json.load(f)

        self.labels = np.unique(self.meta['image_labels']).tolist()
        self.filenames = []
        self.labellist = []
        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.filenames.append(x)
            self.labellist.append(y)

    def __getitem__(self, i):
        key = self.labellist[i]
        im_path = self.filenames[i]
        img = Image.open(im_path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(key)

    def __len__(self):
        return len(self.filenames)


def get_simple_loader(meta_file_path, image_size, batch_size, num_workers, augmentation):
    transform = TransformLoader(image_size=image_size, augmentation=augmentation)
    dataset = DatasetIM(meta_file_path=meta_file_path, transform=transform.get_transform())
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True, shuffle=True)
    return data_loader


if __name__ == '__main__':
    import os
    cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_file = os.path.join(cur_dir, "datasets", "miniimagenet", "base.json")

    dl = get_simple_loader(meta_file, image_size=84, batch_size=64, num_workers=4, augmentation=False)

    for i, (x, y) in enumerate(dl):
        assert x.shape == torch.Size([64, 3, 84, 84])
        assert y.shape == torch.Size([64])
        assert y.max() <= 64


