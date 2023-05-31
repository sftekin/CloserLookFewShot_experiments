import os
import torch
import json
import numpy as np
import torch.nn
import time
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
import backbone
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
import argparse
from data.datamgr import SimpleDataManager, SetDataManager
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
import backbone
from methods.deep_emd import DeepEMD
from methods.emd_utils import emd_load_model, get_deep_emd_args, deep_emd_episode
from methods.simpleshot_utils import ss_step, ss_episode
import methods.ss_backbones as ss_backbones


class DatasetIM:
    def __init__(self, meta_file_path, imsize):
        with open(meta_file_path, 'r') as f:
            self.meta = json.load(f)

        self.labels = np.unique(self.meta['image_labels']).tolist()
        self.filenames = []
        self.labellist = []
        self.counter = {}
        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            if y not in self.counter.keys():
                self.counter[y] = 0
            elif self.counter[y] >= 100:
                continue
            self.filenames.append(x)
            self.labellist.append(y)
            self.counter[y] += 1

        self.norm_param = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }

        self.transform = transforms.Compose([
            transforms.Resize([int(imsize * 1.15), int(imsize * 1.15)]),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
            transforms.Normalize(**self.norm_param)
        ])

    def __getitem__(self, i):
        key = self.labellist[i]
        im_path = self.filenames[i]
        img = Image.open(im_path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(key)

    def __len__(self):
        return len(self.filenames)


def load_model(dataset_name, model_name, method):
    checkpoint_dir = 'checkpoints/%s/%s_%s' % (dataset_name, model_name, method)
    checkpoint_dir += '_%dway_%dshot' % (n_way, n_shot)

    if method in ['relationnet', 'relationnet_softmax']:
        if 'Conv4' in model_name:
            feature_model = backbone.Conv4NP
        elif 'Conv6' in model_name:
            feature_model = backbone.Conv6NP
        else:
            feature_model = lambda: model_dict[model_name](flatten=False)
        loss_type = 'mse' if method == 'relationnet' else 'softmax'
        model = RelationNet(feature_model, loss_type=loss_type, n_way=n_way, n_support=n_shot)
    elif method in ['maml', 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(model_dict[model_name], approx=(method == 'maml_approx'), n_way=n_way, n_support=n_shot)
    elif method == "protonet":
        model = ProtoNet(model_dict[model_name], n_way=n_way, n_support=n_shot)
    elif method == "matchingnet":
        model = MatchingNet(model_dict[model_name], n_way=n_way, n_support=n_shot)
    elif method == "DeepEMD":
        deep_emd_args = get_deep_emd_args(way=n_way, shot=n_shot, query=n_query)
        model = DeepEMD(args=deep_emd_args)
        model = model.cuda()
    elif "simpleshot" in method:
        bb_model = model_name
        bb_mapper = {
            "conv4": ss_backbones.conv4,
            "resnet10": ss_backbones.resnet10,
            "resnet18": ss_backbones.resnet18,
            "resnet34": ss_backbones.resnet34,
            "resnet50": ss_backbones.resnet50,
            "wideres": ss_backbones.wideres,
            "densenet121": ss_backbones.densenet121
        }

        model = bb_mapper[bb_model](num_classes=64, remove_linear=False)
        model = torch.nn.DataParallel(model).cuda()
    else:
        raise ValueError

    if method == "DeepEMD":
        model = emd_load_model(model, dir=f"checkpoints"
                                          f"/{dataset_name}/DeepEMD/max_acc.pth", mode="cuda")
    elif "simpleshot" in method:
        save_path = f"checkpoints/{dataset_name}/SimpleShot/{bb_model}/checkpoint.pth.tar"
        tmp = torch.load(save_path)
        model.load_state_dict(tmp["state_dict"])
    else:
        modelfile = get_best_file(checkpoint_dir)
        model = model.cuda()
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        model.eval()

    return model


if __name__ == '__main__':
    n_query = 15
    n_way = 5
    n_shot = 1
    batch_size = 64
    dataset_type = "val"

    parser = argparse.ArgumentParser(description='few-shot inference')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--method', default='DeepEMD',
                        choices=["maml_approx", "matchingnet", "protonet", "relationnet_softmax", "DeepEMD",
                                 "simpleshot"])
    parser.add_argument('--model_name', default="ResNet18", choices=['Conv4', 'Conv6', 'ResNet10', 'ResNet18',
                                                                     'ResNet34', 'ResNet50', "resnet18"])
    parser.add_argument('--data_set', default="base", choices=["base", "val", "novel"])
    parser.add_argument('--ep_num', default=1000, type=int)
    args = parser.parse_args()
    print(vars(args))

    model = load_model(dataset_name="miniImagenet", model_name=args.model_name, method=args.method)
    model.cuda()

    if args.method == "DeepEMD":
        image_size = 84
    elif "simpleshot" in args.method:
        if "conv4" in args.model_name:
            image_size = 84
        else:
            image_size = 96
    else:
        if "Conv" in args.model_name:
            image_size = 84
        else:
            image_size = 224

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    meta_file = os.path.join(cur_dir, "filelists", "miniImagenet", f"{dataset_type}.json")
    ds = DatasetIM(meta_file, imsize=image_size)

    first_loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    second_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model.n_way = 1
    model.n_support = 1

    distance_mat = np.zeros((len(first_loader), len(first_loader)))
    for i, (sup, sup_y) in enumerate(first_loader):
        # if i == 1: break
        start_time = time.time()
        for j, (query, query_y) in enumerate(second_loader):
            if j % 5 == 0:
                print(f"\r{j / len(second_loader) * 100}", flush=True, end="")
            row_idx = (batch_size * j)
            with torch.no_grad():
                if args.method == "DeepEMD":
                    dist, embed = deep_emd_episode(model, torch.cat([sup, query]).unsqueeze(0), y=None, n_way=1, n_support=1, n_query=query.shape[0])
                else:
                    model.n_query = query.shape[0]
                    dist, _ = model.set_forward(torch.cat([sup, query]).unsqueeze(0))

                distance_mat[i, row_idx:row_idx+batch_size] = dist.cpu().numpy().flatten()
        print(f"\n*** {i}/{len(first_loader)} sample is finished in {time.time() - start_time}***")

    with open("dist.npy", "wb") as f:
        np.save(f, distance_mat)

    #
    # embed_list, label_list = [], []
    # for i, (x, y) in enumerate(data_loader):
    #     print(f"\r{i / len(data_loader) * 100}", flush=True, end="")
    #     # if i == 10: break
    #     if args.method == "DeepEMD":
    #         with torch.no_grad():
    #             scores, embed = deep_emd_episode(model, x, y=None, n_way=n_way, n_support=n_shot, n_query=n_query)
    #             # logits[i, :] = scores.detach().cpu().numpy()
    #     # elif "simpleshot" in args.method:
    #     #     with torch.no_grad():
    #     #         pred, distance, embed = ss_episode(model, x, n_way, n_shot, n_query, out_mean=kwargs["base_mean"])
    #     #         logits[i, :] = distance.T
    #     else:
    #         model.n_query = n_query
    #         scores, embed = model.set_forward(x)
    #         # logits[i, :] = scores.detach().cpu().numpy()
    #
    #
    #     embed_list.append(embed)
    #     label_list.append(y)
    #
    # embed_list = torch.cat(embed_list)
    # label_list = torch.cat(label_list)
    #
    # file_dir = os.path.join(os.path.dirname(__file__), f"features/embeddings/{args.method}/val")
    # if not os.path.exists(file_dir):
    #     os.makedirs(file_dir)
    # s_path = os.path.join(file_dir, f"embed.pt")
    # y_path = os.path.join(file_dir, f"label.pt")
    # torch.save(embed_list, s_path)
    # torch.save(label_list, y_path)
