import numpy as np
import torch
from torch.autograd import Variable

import configs
from methods.ResNet import resnet18
from io_utils import model_dict
from data.datamgr import SimpleDataManager, SetDataManager


def run():
    # Load model
    encoder_path = "checkpoints/miniImagenet/SimpleShot/resnet18/checkpoint.pth.tar"
    encoder_params = torch.load(encoder_path)
    encoder = resnet18(num_classes=64, remove_linear=False)
    encoder = torch.nn.DataParallel(encoder).cuda()
    encoder.load_state_dict(encoder_params["state_dict"])

    # load data
    dataset_name = "miniImagenet"
    n_query = 15
    n_way = 5
    n_shot = 1
    model_name = "ResNet18"
    image_size = 84
    base_file = configs.data_dir[dataset_name] + 'base.json'
    novel_file = configs.data_dir[dataset_name] + 'novel.json'
    data_mgr = SetDataManager(image_size, n_query=n_query, n_way=n_way, n_support=n_shot, n_eposide=100)

    base_loader = data_mgr.get_data_loader(base_file, aug=False)
    novel_loader = data_mgr.get_data_loader(novel_file, aug=False)

    out_mean = []
    for i, (x, y) in enumerate(base_loader):
        print(f"\rTrain Episode {i} / {len(base_loader)}", end="", flush=True)
        with torch.no_grad():
            x = Variable(x.cuda())
            x = x.contiguous().view(n_way * (n_shot + n_query), *x.size()[2:])
            output, fc_output = encoder(x, True)
            out_mean.append(output.cpu().data.numpy())
    out_mean = np.concatenate(out_mean, axis=0).mean(0)

    acc_all = []
    y_query = np.repeat(range(n_way), n_query)
    for i, (x, y) in enumerate(novel_loader):
        print(f"\rNovel Episode {i} / {len(novel_loader)}", end="", flush=True)
        with torch.no_grad():
            x = Variable(x.cuda())
            x = x.contiguous().view(n_way * (n_shot + n_query), *x.size()[2:])
            output, fc_output = encoder(x, True)
            output = output.view(n_way, n_shot + n_query, -1)
            support = output[:, :n_shot]
            query = output[:, n_shot:]

            support = support.contiguous().mean(1)
            query = query.contiguous().view(n_way * n_query, -1)

            pred = metric_class_type(support.cpu().numpy(),
                                     query.cpu().numpy(),
                                     base_mean=out_mean, k=1)

            corrects = np.sum(pred == y_query)
            acc = corrects / len(y_query) * 100
            acc_all.append(acc)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print(f'Acc = {acc_mean:.2f} +- {1.96 * acc_std / np.sqrt(len(novel_loader)):.2f}')


def metric_class_type(support, query, base_mean, k=1):
    support -= base_mean
    support /= np.linalg.norm(support, 2, 1)[:, None]

    query -= base_mean
    query /= np.linalg.norm(query, 2, 1)[:, None]

    subtract = support[:, None, :] - query
    distance = np.linalg.norm(subtract, 2, axis=-1)

    idx = np.argpartition(distance, k, axis=0)[:k]
    return idx


if __name__ == '__main__':
    run()

