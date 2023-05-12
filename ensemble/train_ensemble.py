import os
import glob
import itertools
import numpy as np
import scipy
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=input_dim),
            nn.Linear(input_dim, hidden_dim[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_dim[1], output_dim),
            nn.Sigmoid()
        )
        self.net.apply(self.init_weights)

    def forward(self, x):
        out = self.net(x)
        out = torch.softmax(out, dim=-1)
        return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


def min_max(in_arr):
    min_ = in_arr.min()
    max_ = in_arr.max()
    return (in_arr - min_) / (max_ - min_)


def standardize(in_arr):
    return (in_arr - in_arr.mean()) / in_arr.std()


def load_logits(model_names, dataset, perform_norm=True):
    save_dir = "model_outs"
    logits = []
    for model_n in model_names:
        logit = np.load(f"{save_dir}/{model_n}_{dataset}_logits.npy")
        if model_n == "DeepEMD":
            logit = np.transpose(logit.reshape(1000, 15, 5, 5), axes=[0, 2, 1, 3])
            logit = logit.reshape(1000, 75, 5)
        logits.append(logit.reshape(-1, 5))

    if perform_norm:
        logits_t = []
        for l in range(len(model_names)):
            logit_ = min_max(logits[l])
            logits_t.append(scipy.special.softmax(logit_, axis=1))
        logits = logits_t

    return logits


def create_data(logits):
    x = np.concatenate(logits, axis=1)
    y = np.tile(np.repeat(range(n_way), n_query), 1000).flatten()
    data = np.concatenate([x, y[:, None]], axis=1)
    random_idx = np.random.permutation(range(len(data)))
    data = data[random_idx]

    return data


# def get_best_model(check_point_dir):
#     exp_dirs = list(glob.glob(f"{check_point_dir}/exp_*"))
#     best_acc = 0
#     best_dict = {}
#     for exp_dir in exp_dirs:
#         best_model_path = f"{exp_dir}/best_model.tar"
#         if not os.path.exists(best_model_path):
#             continue
#         temp_dict = torch.load(best_model_path)
#         if temp_dict["accuracy"] > best_acc:
#             best_dict = temp_dict
#             best_acc = temp_dict["accuracy"]
#     return best_dict
#
#
# def create_exp_dir():
#     check_point_dir = "ens_checkpoints"
#     if not os.path.exists(check_point_dir):
#         os.makedirs(check_point_dir)
#         exp_num = 1
#     else:
#         exp_dirs = list(glob.glob(f"{check_point_dir}/exp_*"))
#         exp_nums = [int(os.path.basename(exp).split("_")[1]) for exp in exp_dirs]
#         exp_num = sorted(exp_nums)[-1] + 1
#     exp_dir = f"{check_point_dir}/exp_{exp_num}"
#     os.makedirs(exp_dir)
#     return exp_dir


def test_loop(model, data_loader):
    acc_all = []
    for i, batch_data in enumerate(data_loader):
        in_x = batch_data[:, :-1].to("cuda").float()
        scores = model(in_x)
        label = batch_data[:, -1].numpy()

        pred = scores.argmax(dim=1).detach().cpu().numpy()
        corrects = np.sum(pred == label)
        acc_all.append(corrects / len(label) * 100)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)

    return acc_mean, acc_std


def run(model_names, normalize_flag, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_logits = load_logits(model_names, "base", perform_norm=normalize_flag)
    train_data = create_data(train_logits)

    val_logits = load_logits(model_names, "val", perform_norm=normalize_flag)
    val_data = create_data(val_logits)

    novel_logits = load_logits(model_names, "novel", perform_norm=normalize_flag)
    novel_data = create_data(novel_logits)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    novel_loader = DataLoader(novel_data, batch_size=64, shuffle=True)
    model = MLP(len(model_names) * 5, [100, 100], 5)

    model = model.to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0
    for epoch in range(n_epochs):

        avg_loss = []
        for i, batch_data in enumerate(train_dataloader):
            in_x = batch_data[:, :-1].to("cuda").float()
            label = batch_data[:, -1].type(torch.long).to("cuda")

            optimizer.zero_grad()
            out = model(in_x)
            loss = loss_fn(out, label)

            # if lambda_1 > 0:
            #     # get L1 over weights
            #     l1_reg = torch.tensor(0., requires_grad=True).float().to("cuda")
            #     for name, param in model.named_parameters():
            #         if "weight" in name:
            #             l1_reg = l1_reg + torch.norm(param, p=1)
            #     # return regularized loss (L2 is applied with optimizer)
            #     loss = loss + lambda_1 * l1_reg

            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        if epoch % 10 == 0:
            run_loss = np.mean(avg_loss)
            print(f'Epoch {epoch} | Loss {run_loss:.4f}')

        acc_mean, acc_std = test_loop(model, val_loader)

        if acc_mean > best_val_acc:
            conf = 1.96 * acc_std / np.sqrt(len(val_loader))
            print(f'best model Val Acc = {acc_mean:.4f} +- {conf:.2f}')

            outfile = os.path.join(save_dir, f'best_model.tar')
            torch.save({'epoch': epoch,
                        'state': model.state_dict(),
                        "accuracy": acc_mean,
                        "confidence": conf}, outfile)
            best_val_acc = acc_mean

    best_dict = torch.load(f"{save_dir}/best_model.tar")
    model.load_state_dict(best_dict["state"])
    model.eval()

    acc_mean, acc_std = test_loop(model, novel_loader)
    conf = 1.96 * acc_std / np.sqrt(len(novel_loader))
    print(f'Novel Acc = {acc_mean:.4f} +- {conf:.2f}')
    exp_result = dict(val_acc=best_dict["accuracy"],
                      val_conf=best_dict["confidence"],
                      test_acc=acc_mean,
                      test_conf=conf,
                      state=model.state_dict(),
                      model_names=model_names)
    torch.save(exp_result, f"{save_dir}/results.tar")
    print(f"{model_names} finished with Acc = {acc_mean:.4f} +- {conf:.2f}...")


if __name__ == '__main__':
    n_query = 15
    n_way = 5
    n_epochs = 300
    # lambda_1 = 0.01
    # temperatures = [1, 1, 1]
    # temp_flag = True
    normalize = True
    all_names = ["maml_approx", "matchingnet", "protonet", "relationnet_softmax", "DeepEMD", "simpleshot"]
    ens_sizes = np.arange(2, len(all_names))
    for ens_size in ens_sizes:
        combinations = itertools.combinations(all_names, ens_size)
        for comb in combinations:
            sv_path = f"ens_checkpoints/{'-'.join(comb)}"
            run(model_names=comb, normalize_flag=normalize, save_dir=sv_path)
