import os
import time
import configs
import torch
import numpy as np
import random
import tqdm
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


def infer(model, loader, mode, method, model_name, n_query, n_way, n_shot, **kwargs):
    print(f"obtaining {mode} outputs")
    acc_all = []
    logits = np.zeros((len(loader), n_query * n_way, n_way))
    predicts = np.zeros((len(loader), n_query * n_way))
    negatives = np.zeros((len(loader), n_query * n_way))
    start_time = time.time()
    for i, file_name in enumerate(loader):
        with open(file_name, "rb") as f:
            x, y = torch.load(f)
        if method == "DeepEMD":
            with torch.no_grad():
                scores, embed = deep_emd_episode(model, x, y, n_way=n_way, n_support=n_shot, n_query=n_query)
                y_query = np.tile(range(n_way), n_query)
                pred = scores.argmax(dim=1).detach().cpu().numpy()
                logits[i, :] = scores.detach().cpu().numpy()
        elif "simpleshot" in method:
            with torch.no_grad():
                pred, distance, embed = ss_episode(model, x, n_way, n_shot, n_query, out_mean=kwargs["base_mean"])
                logits[i, :] = distance.T
                y_query = np.repeat(range(n_way), n_query)
                pred = pred.squeeze()
        else:
            model.n_query = x.size(1) - n_shot
            scores, embed = model.set_forward(x)
            y_query = np.repeat(range(n_way), model.n_query)
            pred = scores.argmax(dim=1).detach().cpu().numpy()
            logits[i, :] = scores.detach().cpu().numpy()

        if kwargs["save_features"]:
            file_dir = os.path.join(os.path.dirname(__file__), f"features/{method}/{mode}")
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            q_path = os.path.join(file_dir, f"support_{i}.pt")
            s_path = os.path.join(file_dir, f"query_{i}.pt")
            torch.save(embed[0], q_path)
            torch.save(embed[1], s_path)

        predicts[i, :] = pred
        negatives[i, pred != y_query] = 1
        corrects = np.sum(pred == y_query)
        acc = corrects / len(y_query) * 100
        print(f"\rEpisode {i} / {len(loader)}: {acc:.2f}", end="", flush=True)
        acc_all.append(acc)

    epoch_time = time.time() - start_time
    print(f"Took {epoch_time:.2f} seconds")

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print(f'{mode}->{len(loader)} Acc = {acc_mean:.2f} +- {1.96 * acc_std / np.sqrt(len(loader)):.2f}')
    np.save(f"ensemble/model_outs/{method}_{model_name}_{mode}_logits.npy", logits)
    np.save(f"ensemble/model_outs/{method}_{model_name}_{mode}_predicts.npy", predicts)
    np.save(f"ensemble/negatives/{method}_{model_name}_{mode}_negatives.npy", negatives)


def run(method, data_set, ep_num, model_name):
    dataset_name = "miniImagenet"
    n_query = 15
    n_way = 5
    n_shot = 1
    base_file = configs.data_dir[dataset_name] + f'{data_set}.json'

    if method == "DeepEMD":
        image_size = 84
    elif "simpleshot" in method:
        if "conv4" in model_name:
            image_size = 84
        else:
            image_size = 96
    else:
        if "Conv" in model_name:
            image_size = 84
        else:
            image_size = 224

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, dataset_name, model_name, method)
    checkpoint_dir += '_%dway_%dshot' % (n_way, n_shot)

    data_mgr = SetDataManager(image_size, n_query=n_query, n_way=n_way, n_support=n_shot, n_eposide=ep_num)
    loader = data_mgr.get_data_loader(base_file, aug=False)

    # I have to store the loader since during initialization of models,
    # there is a random op. which changes the seed for the next call.
    print(f'fix {data_set} set for all epochs')
    temp_dir = os.path.join("temp", str(image_size), data_set)
    loader_list = []
    if os.path.exists(temp_dir):
        print(f'{temp_dir} found...')
        files = np.array([os.path.join(temp_dir, f) for f in os.listdir(temp_dir)])
        idx = [int(os.path.splitext(f)[0].split("_")[1]) for f in files]
        loader_list = files[np.argsort(idx)].tolist()
    else:
        print(f'{temp_dir} not found creating...')
        os.makedirs(temp_dir)
        for i, batch in enumerate(loader):
            print(f"\rEpisode {i} / {len(loader)}", end="", flush=True)
            file_name = os.path.join(temp_dir, f"batch_{i}.pt")
            loader_list.append(file_name)
            with open(file_name, "wb") as f:
                torch.save(batch, f)

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
        model = emd_load_model(model, dir=f"{configs.save_dir}/checkpoints"
                                          f"/{dataset_name}/DeepEMD/max_acc.pth", mode="cuda")
    elif "simpleshot" in method:
        save_path = f"{configs.save_dir}/checkpoints/{dataset_name}/SimpleShot/{bb_model}/checkpoint.pth.tar"
        tmp = torch.load(save_path)
        model.load_state_dict(tmp["state_dict"])
    else:
        modelfile = get_best_file(checkpoint_dir)
        model = model.cuda()
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        model.eval()

    # prep infer args
    infer_args = {
        "model": model,
        "loader": loader_list,
        "method": method,
        "model_name": model_name,
        "mode": data_set,
        "n_query": n_query,
        "n_way": n_way,
        "n_shot": n_shot,
        "save_features": False
    }

    if "simpleshot" in method:
        print("Simple shot requires the mean of the features extracted from base dataset")
        base_ds_path = configs.data_dir[dataset_name] + f'base.json'
        base_loader = data_mgr.get_data_loader(base_ds_path, aug=False)
        base_mean = []
        with torch.no_grad():
            for i, (x, y) in enumerate(base_loader):
                print(f"\rBase Episode {i} / {len(base_loader)}", end="", flush=True)
                output, fc_output = ss_step(model, x, n_way, n_shot, n_query)
                base_mean.append(output.detach().cpu().data.numpy())
        base_mean = np.concatenate(base_mean, axis=0).mean(0)
        infer_args["base_mean"] = base_mean

    # train
    infer(**infer_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='few-shot inference')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--method', default='protonet',
                        choices=["maml_approx", "matchingnet", "protonet", "relationnet_softmax", "DeepEMD",
                                 "simpleshot"])
    parser.add_argument('--model_name', default="ResNet18", choices=['Conv4', 'Conv6', 'ResNet10', 'ResNet18',
                                                                     'ResNet34', 'ResNet50', "resnet18"])
    parser.add_argument('--data_set', default="base", choices=["base", "val", "novel"])
    parser.add_argument('--ep_num', default=1000, type=int)
    args = parser.parse_args()
    print(vars(args))

    # set the seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run(method=args.method,
        data_set=args.data_set,
        ep_num=args.ep_num,
        model_name=args.model_name)
