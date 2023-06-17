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
from data_manager.episode_loader import get_episode_loader
import methods.ss_backbones as ss_backbones
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def infer(model, loader, mode, method, model_name, n_query, n_way, n_shot, **kwargs):
    print(f"obtaining {mode} outputs")
    acc_all = []
    logits = np.zeros((len(loader), n_query * n_way, n_way))
    predicts = np.zeros((len(loader), n_query * n_way))
    negatives = np.zeros((len(loader), n_query * n_way))
    one_forward_pass = 0
    for i, (x, y) in enumerate(loader):
        if method == "DeepEMD":
            with torch.no_grad():
                start_time = time.time()
                scores = deep_emd_episode(model, x, y, n_way=n_way, n_support=n_shot, n_query=n_query)
                one_forward_pass += time.time() - start_time
                y_query = np.tile(range(n_way), n_query)
                pred = scores.argmax(dim=1).detach().cpu().numpy()
                logits[i, :] = scores.detach().cpu().numpy()
        elif "simpleshot" in method:
            with torch.no_grad():
                start_time = time.time()
                pred, distance = ss_episode(model, x, n_way, n_shot, n_query, out_mean=kwargs["base_mean"])
                one_forward_pass += time.time() - start_time
                logits[i, :] = distance.T
                y_query = np.repeat(range(n_way), n_query)
                pred = pred.squeeze()
        else:
            model.n_query = x.size(1) - n_shot
            start_time = time.time()
            scores = model.set_forward(x)
            one_forward_pass += time.time() - start_time
            y_query = np.repeat(range(n_way), model.n_query)
            pred = scores.data.cpu().numpy().argmax(axis=1)
            logits[i, :] = scores.detach().cpu().numpy()

        predicts[i, :] = pred
        negatives[i, pred != y_query] = 1
        corrects = np.sum(pred == y_query)
        acc = corrects / len(y_query) * 100
        print(f"\rEpisode {i + 1} / {len(loader)}: {acc:.2f}", end="", flush=True)
        acc_all.append(acc)

    print(f"\nApprox, one forward pass takes {one_forward_pass / len(loader):.3f} seconds")

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print(f'{mode}->{len(loader)} Acc = {acc_mean:.2f} +- {1.96 * acc_std / np.sqrt(len(loader)):.2f}')

    model_outs_dir = os.path.join(configs.save_dir, "inference", kwargs["dataset_name"], "model_outs", method)
    if not os.path.exists(model_outs_dir):
        os.makedirs(model_outs_dir)

    save_str = f"{method}_{model_name}_{mode}" if method != "DeepEMD" else f"{method}_{mode}"
    np.save(os.path.join(model_outs_dir, f"{save_str}_logits.npy"), logits)
    np.save(os.path.join(model_outs_dir, f"{save_str}_predicts.npy"), predicts)
    np.save(os.path.join(model_outs_dir, f"{save_str}_negatives.npy"), negatives)


def calc_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))


def run(method, dataset_name, class_type, ep_num, model_name, aug_used=False, cross=False):
    n_query = 15
    n_way = 5
    n_shot = 1
    base_file = configs.data_dir[dataset_name] + f'{class_type}.json'

    if method == "DeepEMD":
        image_size = 84
    elif "simpleshot" in method:
        if model_name.lower() in ["conv4", "conv6"]:
            image_size = 84
        else:
            image_size = 96
    else:
        if "Conv" in model_name:
            image_size = 84
        else:
            image_size = 224

    loader = get_episode_loader(meta_file_path=base_file, image_size=image_size, n_episodes=ep_num,
                                augmentation=False, n_way=n_way, n_shot=n_shot, n_query=n_query,
                                num_workers=8, load_sampler_indexes=True, dataset_name=dataset_name)

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
        bb_model = model_name.lower()
        bb_mapper = {
            "conv4": ss_backbones.conv4,
            "conv6": ss_backbones.conv6,
            "resnet10": ss_backbones.resnet10,
            "resnet18": ss_backbones.resnet18,
            "resnet34": ss_backbones.resnet34,
            "resnet50": ss_backbones.resnet50,
            "wideres": ss_backbones.wideres,
            "densenet121": ss_backbones.densenet121
        }
        num_classes = 100 if dataset_name == "CUB" else 64
        model = bb_mapper[bb_model](num_classes=num_classes, remove_linear=False)
        model = torch.nn.DataParallel(model).cuda()
    else:
        raise ValueError

    if cross:
        trained_dataset = "miniImagenet"
    else:
        trained_dataset = dataset_name

    if method == "DeepEMD":
        model = emd_load_model(model, dir=f"{configs.save_dir}/checkpoints"
                                          f"/{trained_dataset}/DeepEMD/max_acc.pth", mode="cuda")
    elif "simpleshot" in method:
        save_path = f"{configs.save_dir}/checkpoints/{trained_dataset}/SimpleShot/{bb_model}/checkpoint.pth.tar"
        tmp = torch.load(save_path)
        model.load_state_dict(tmp["state_dict"])
    else:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, trained_dataset, model_name, method)
        if aug_used:
            checkpoint_dir += "_aug"
        checkpoint_dir += '_%dway_%dshot' % (n_way, n_shot)

        modelfile = get_best_file(checkpoint_dir)

        model = model.cuda()
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        # model.eval()

    # prep infer args
    infer_args = {
        "model": model,
        "loader": loader,
        "method": method,
        "model_name": model_name,
        "mode": class_type,
        "n_query": n_query,
        "n_way": n_way,
        "n_shot": n_shot,
        "save_features": False,
        "dataset_name": "cross" if cross else dataset_name
    }

    calc_model_size(model)

    if "simpleshot" in method:
        print("Simple shot requires the mean of the features extracted from base dataset")
        base_ds_path = configs.data_dir[dataset_name] + f'base.json'
        base_loader = get_episode_loader(meta_file_path=base_ds_path, image_size=image_size, n_episodes=ep_num,
                                         augmentation=False, n_way=n_way, n_shot=n_shot, n_query=n_query,
                                         num_workers=8, load_sampler_indexes=True, dataset_name=dataset_name)
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
    parser.add_argument('--seed', default=50, type=int)
    parser.add_argument('--dataset_name', default="CUB", choices=["CUB", "miniImagenet"])
    parser.add_argument('--method', default='DeepEMD',
                        choices=["maml_approx", "matchingnet", "protonet", "relationnet",
                                 "relationnet_softmax", "DeepEMD",
                                 "simpleshot"])
    parser.add_argument('--model_name', default="ResNet18", choices=['Conv4', 'Conv6', 'ResNet10', 'ResNet18',
                                                                        'ResNet34', "WideRes", "DenseNet121"])
    parser.add_argument('--class_type', default="novel", choices=["base", "val", "novel"])
    parser.add_argument('--ep_num', default=600, type=int)
    parser.add_argument('--aug_used', action='store_true',  help='performed train augmentation')
    parser.add_argument('--cross', action='store_true')
    args = parser.parse_args()
    print(vars(args))

    # set the seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run(method=args.method,
        dataset_name=args.dataset_name,
        class_type=args.class_type,
        ep_num=args.ep_num,
        model_name=args.model_name,
        aug_used=args.aug_used,
        cross=args.cross)
