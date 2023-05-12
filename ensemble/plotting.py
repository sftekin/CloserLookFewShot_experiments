import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score, cohen_kappa_score, accuracy_score
from helper import load_predictions, calculate_errors, calc_perf
from diversity_stats import calc_stat_matrices, calc_generalized_div
from ensemble_methods import ensemble_methods


def plot_acc(mode="train"):
    model_acc = {}
    for model_n in model_names:
        preds = np.load(f"model_outs/{model_n}_{mode}_predicts.npy")
        acc = []
        for i in range(preds.shape[0]):
            y = np.repeat(range(n_way), n_query) if model_n != "DeepEMD" else np.tile(range(n_way), n_query)
            acc.append(np.sum(preds[i] == y) / (n_way * n_query))
        model_acc[model_n] = acc.copy()

    plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize=(15, 8))
    for model_n, acc in model_acc.items():
        ax.plot(acc, label=model_n)

    ax.set_xticks(np.arange(0, preds.shape[0] + 1, preds.shape[0] // 10), fontsize=12)
    ax.set_xlabel("n_episodes", fontsize=12)
    ax.set_ylabel("acc", fontsize=12)
    ax.set_title(f"{mode} Accuracies for each Episode", fontsize=12)
    ax.legend(fontsize=12, fancybox=True, framealpha=0.5)
    plt.savefig(f"{mode}_acc.png", dpi=200)


def get_topk_acc(k, mode="train"):
    model_acc = {}
    for model_n in model_names:
        preds = np.load(f"model_outs/{model_n}_{mode}_logits.npy")
        acc = []
        for i in range(preds.shape[0]):
            y = np.repeat(range(n_way), n_query) if model_n != "DeepEMD" else np.tile(range(n_way), n_query)
            top_k = top_k_accuracy_score(y, preds[i], k=k) * 100
            acc.append(top_k)
        acc = np.asarray(acc)
        acc_mean = np.mean(acc)
        acc_std = np.std(acc)
        ci = 1.96 * acc_std / np.sqrt(preds.shape[0])
        acc_str = f"{acc_mean:.2f} +- {ci:.2f}"
        model_acc[model_n] = acc_str
    return model_acc


def plot_statistics(mode):
    predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_way=n_way, set_name=mode)
    errors, err_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)
    stats = calc_stat_matrices(errors)
    for stat_name, stat_df in stats.items():
        stat_path = os.path.join(figures_dir, f"{stat_name}.png")
        plot_heatmap(stat_df, title=f"Set:{mode}, Stat:{stat_name}", save_path=stat_path)


def plot_focal_div(mode, ens_mode="avg", drop_nans=False):
    predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_way=n_way, set_name=mode)
    all_error_dict, all_error_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)

    ens_dict = {}
    ens_sizes = np.arange(2, len(model_names))
    for ens_size in ens_sizes:
        focal_div_dict = {}
        combinations = itertools.combinations(range(len(model_names)), ens_size)
        for comb in combinations:
            # select ensemble set
            set_bin_arr = all_error_arr[:, comb]
            set_preds = pred_arr[:, comb]

            # calc focal diversity
            focal_div = 0
            for focal_idx in comb:
                focal_arr = all_error_dict[model_names[focal_idx]]
                neg_idx = np.where(focal_arr == 0)[0]
                neg_samp_arr = set_bin_arr[neg_idx]
                focal_div += calc_generalized_div(neg_samp_arr)
            focal_div /= ens_size

            # calculate accuracy of ensemble
            if ens_mode == "avg":
                all_acc = set_bin_arr.sum(axis=0) / len(set_bin_arr)
                all_acc = all_acc.mean()
            elif ens_mode == "plurality":
                ens_pred = ensemble_methods["voting"](set_preds, method="plurality",
                                                      n_way=n_way, n_query=n_query)

                ens_bin_arr = calc_perf(ens_pred.flatten(), n_query=n_query, n_way=n_way)
                all_acc = ens_bin_arr.sum(axis=0) / len(ens_bin_arr)
            else:
                ens_pred = ensemble_methods["voting"](set_preds, method="majority",
                                                      n_way=n_way, n_query=n_query)
                acc_score = []
                y = np.repeat(range(n_way), n_query)
                for i in range(len(ens_pred)):
                    nan_vals = np.isnan(ens_pred[i])
                    if nan_vals.any():
                        if drop_nans:
                            acc = accuracy_score(y[~nan_vals], ens_pred[i][~nan_vals]) * 100
                        else:
                            acc = np.mean(y == ens_pred) * 100
                    else:
                        acc = accuracy_score(y, ens_pred[i]) * 100
                    acc_score.append(acc)
                all_acc = np.array(acc_score).mean()

            # add to collection
            focal_div_dict[comb] = [focal_div, all_acc]

        # add to ens collection
        ens_dict[ens_size] = focal_div_dict

    fig, ax = plt.subplots(figsize=(10, 8))
    for s, focal_div in ens_dict.items():
        save_path = os.path.join(figures_dir, f"{mode}_focal_div_{s}_{ens_mode}_{drop_nans}.png")
        acc_div_arr = np.array(list(focal_div.values()))
        ax.scatter(acc_div_arr[:, 0], acc_div_arr[:, 1])
        min_val, max_val = acc_div_arr.min(axis=0) - 0.01, acc_div_arr.max(axis=0) + 0.01
        x_ticks = np.linspace(min_val[0], max_val[0], 10)
        y_ticks = np.linspace(min_val[1], max_val[1], 10)
        x_tick_labels = [f"{i:.2f}" for i in x_ticks]
        y_tick_labels = [f"{i:.2f}" for i in y_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("focal div")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"Ensemble size: {s}, Accuracy mode: {ens_mode}, nan drop: {drop_nans}, dataset={mode}")
        ax.grid()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.cla()

    all_ens = {}
    for ens_set, foc_div_dicts in ens_dict.items():
        for k, v in foc_div_dicts.items():
            all_ens[k] = v
    all_scores = np.array(list(all_ens.values()))
    sorted_idx = np.argsort(all_scores[:, 1])
    sorted_all_scores = all_scores[sorted_idx]
    topk = 20
    topk_scores = sorted_all_scores[-topk:]
    topk_ens = [list(all_ens.keys())[i] for i in sorted_idx[-topk:]]

    fig, ax = plt.subplots(figsize=(10, 8))
    save_path = os.path.join(figures_dir, f"{mode}_focal_div_all_{ens_mode}_{drop_nans}.png")
    total_size = 0
    for s, focal_div in ens_dict.items():
        acc_div_arr = np.array(list(focal_div.values()))
        ax.scatter(acc_div_arr[:, 0], acc_div_arr[:, 1], label=f"ens_size={s}")
        total_size += len(acc_div_arr)
    for i in range(topk):
        ax.text(topk_scores[i, 0], topk_scores[i, 1], topk_ens[i])
    ax.set_xlabel("focal div")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Total Ens: {total_size}, Accuracy mode: {ens_mode}, drop nans: {drop_nans}, dataset={mode}")
    ax.grid()
    ax.legend()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")


def plot_negatives(mode):
    predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_way=n_way, set_name=mode)
    errors, err_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)
    neg_acc = np.zeros((len(model_names), len(model_names)))
    for i, fir_n in enumerate(model_names):
        fir_arr = errors[fir_n]
        where_wrong = np.where(fir_arr == 0)[0]
        for j, second_n in enumerate(model_names):
            sec_arr_pred = errors[second_n][where_wrong]
            neg_acc[i, j] = (sec_arr_pred.sum() / len(sec_arr_pred)) * 100
    stat_path = os.path.join(figures_dir, f"cross_acc.png")
    cross_df = pd.DataFrame(neg_acc, columns=model_names, index=model_names)
    plot_heatmap(cross_df, title=f"Accuracy of other models in the negative samples of a model", save_path=stat_path)


def plot_heatmap(in_df, title, save_path):
    plt.figure()
    sns.heatmap(in_df, annot=True, cmap="Reds", fmt=".2f")
    plt.title(title)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")


if __name__ == '__main__':
    n_query = 15
    n_way = 5
    model_names = ["maml_approx", "matchingnet", "protonet", "relationnet_softmax", "DeepEMD", "simpleshot"]
    figures_dir = "figures"
    ds_name = "novel"

    # plot_statistics(mode="val")
    # plot_negatives(mode=ds_name)
    plot_focal_div(mode=ds_name, ens_mode="majority", drop_nans=False)

    # plot_acc(n_query, n_way, mode="train")
    # plot_acc(n_query, n_way, mode="val")
    # plot_acc(n_query, n_way, mode="test")
    # sets = ["val", "test"]
    # for set_n in sets:
    #     print(set_n)
    #     top1_dict = get_topk_acc(n_query, n_way, model_names, mode=set_n, k=1)
    #     top2_dict = get_topk_acc(n_query, n_way, model_names, mode=set_n, k=2)
    #     top3_dict = get_topk_acc(n_query, n_way, model_names, mode=set_n, k=3)
    #     print(top1_dict)
    #     print(top2_dict)
    #     print(top3_dict)

    # plot_negatives(mode="novel", fig_dir=figures_dir)
