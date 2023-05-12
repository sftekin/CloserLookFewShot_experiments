import os
import itertools
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score, cohen_kappa_score, accuracy_score
from helper import load_predictions, calculate_errors, calc_perf, load_all_degree_preds
from diversity_stats import calc_stat_matrices, calc_generalized_div
from ensemble_methods import ensemble_methods
import scipy


def calc_entropy(in_row):
    count = np.bincount(in_row.astype(int))
    return scipy.stats.entropy(count)


def ens_analysis(mode):
    predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_way=n_way, set_name=mode)
    # deg_preds = load_all_degree_preds(model_names=model_names, n_query=n_query, n_way=n_way, set_name=mode)
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

            # # calculate mean accuracy
            # mean_acc = 0
            # for i in range(len(comb)):
            #     mean_acc += set_bin_arr[:, i].mean() * 100
            # mean_acc /= len(comb)

            # entropy based uncertainty
            # uncertainty = np.apply_along_axis(arr=set_preds, func1d=calc_entropy, axis=1).mean()

            # calculate accuracy of ensemble
            ens_pred = ensemble_methods["voting"](set_preds, method="plurality",
                                                  n_way=n_way, n_query=n_query)
            y = np.tile(np.repeat(range(n_way), n_query), len(ens_pred))
            ens_pred_flatten = ens_pred.flatten()
            all_acc = np.mean(y == ens_pred_flatten) * 100

            # calculate the uncertainty of ensemble
            mjr_pred = ensemble_methods["voting"](set_preds, method="majority", n_way=n_way, n_query=n_query)
            mjr_pred_flatten = mjr_pred.flatten()
            nan_idx = np.isnan(mjr_pred_flatten)
            acc_wt_nans = np.mean(y == mjr_pred_flatten) * 100
            acc_wo_nans = accuracy_score(y[~nan_idx], mjr_pred_flatten[~nan_idx]) * 100
            uncertainty = acc_wo_nans - acc_wt_nans

            # add to collection
            focal_div_dict[comb] = [focal_div, uncertainty, all_acc]

        # add to ens collection
        ens_dict[ens_size] = focal_div_dict

    with open(f"ens_dict_{mode}.pkl", "wb") as f:
        pkl.dump(ens_dict, f)

    # plot acc vs div
    ens_fig_dir = os.path.join(figures_dir, "ens_analysis")
    if not os.path.exists(ens_fig_dir):
        os.makedirs(ens_fig_dir)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for s, focal_div in ens_dict.items():
        stats_arr = np.array(list(focal_div.values()))
        ax[0].scatter(stats_arr[:, 0], stats_arr[:, 2], label=f"ens_size={s}")
        ax[1].scatter(stats_arr[:, 1], stats_arr[:, 2], label=f"ens_size={s}")
        ax[2].scatter(stats_arr[:, 0], stats_arr[:, 1], label=f"ens_size={s}")

    for i in range(3):
        ax[i].grid()
        ax[i].legend()
    ax[0].set_xlabel("Focal Diversity")
    ax[0].set_ylabel("Accuracy (%)")

    ax[1].set_xlabel("Uncertainty")
    ax[1].set_ylabel("Accuracy (%)")

    ax[2].set_xlabel("Focal Diversity")
    ax[2].set_ylabel("Uncertainty")

    all_ens = {}
    for ens_set, foc_div_dicts in ens_dict.items():
        for k, v in foc_div_dicts.items():
            all_ens[k] = v

    for k, v in all_ens.items():
        ax[0].text(v[0], v[2], k, fontsize=8)
        ax[1].text(v[1], v[2], k, fontsize=8)
        ax[2].text(v[0], v[1], k, fontsize=8)

    plt.suptitle(f"Ensemble Analysis, dataset={mode}")
    save_path = os.path.join(ens_fig_dir, f"{mode}_acc_uncertainty_div.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")


def prune_ens(mode, prune_per=0.1):
    with open(f"ens_dict_{mode}.pkl", "rb") as f:
        ens_dict = pkl.load(f)

    all_ens = {}
    for ens_set, foc_div_dicts in ens_dict.items():
        for k, v in foc_div_dicts.items():
            all_ens[k] = v

    def min_max(in_arr):
        min_ = in_arr.min()
        max_ = in_arr.max()
        return (in_arr - min_) / (max_ - min_)

    all_ens_arr = np.array(list(all_ens.values()))
    norm_arr = np.apply_along_axis(arr=all_ens_arr, func1d=min_max, axis=0)
    scores = norm_arr[:, 2] + norm_arr[:, 0] - norm_arr[:, 1]
    num_k = int(len(scores) * prune_per)
    sort_idx = np.argsort(scores)[::-1][:num_k]
    decision_arr = all_ens_arr[sort_idx]

    ens_fig_dir = os.path.join(figures_dir, "ens_analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(all_ens_arr[:, 0], all_ens_arr[:, 2], c="navy", label="pruned")
    ax[0].scatter(decision_arr[:, 0], decision_arr[:, 2], c="r", label="selected")
    ax[0].set_xlabel("focal diversity")
    ax[0].set_ylabel("Accuracy (%)")
    ax[0].set_title(f"Selecting based on diversity")
    ax[0].legend()
    ax[0].grid()
    selected_ens_k = [k for i, k in enumerate(all_ens.keys()) if i in sort_idx]
    for k in selected_ens_k:
        v = all_ens[k]
        ax[0].text(v[0], v[2], k)

    ax[1].scatter(all_ens_arr[:, 1], all_ens_arr[:, 2], c="navy", label="pruned")
    ax[1].scatter(decision_arr[:, 1], decision_arr[:, 2], c="r", label="selected")
    ax[1].set_xlabel("Uncertainty")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title(f"Selecting based on uncertainty")
    ax[1].legend()
    ax[1].grid()
    selected_ens_k = [k for i, k in enumerate(all_ens.keys()) if i in sort_idx]
    for k in selected_ens_k:
        v = all_ens[k]
        ax[1].text(v[1], v[2], k)
    plt.suptitle(f"Pruning Ensembles of {mode}, k={num_k}")
    plt.savefig(os.path.join(ens_fig_dir, f"{mode}_pruned_ens_{num_k}.png"), dpi=200, bbox_inches="tight")


if __name__ == '__main__':
    n_query = 15
    n_way = 5
    model_names = ["maml_approx", "matchingnet", "protonet", "relationnet_softmax", "DeepEMD", "simpleshot"]
    figures_dir = "figures"
    ds_name = "val"

    ens_analysis(mode=ds_name)
    # prune_ens(ds_name, prune_per=0.3)
