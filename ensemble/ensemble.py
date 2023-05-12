import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.metrics import accuracy_score
from ensemble_methods import ensemble_methods
from helper import load_predictions
import itertools


def run():
    model_names = ["maml_approx", "matchingnet", "protonet", "relationnet_softmax", "DeepEMD", "simpleshot"]
    model_colors = {mn: f"C{i}" for i, mn in enumerate(model_names)}
    ens_colors = {
        "ensemble_majority": "maroon",
        "ensemble_plurality": "firebrick"
    }

    comb_acc = {}
    nan_comb = {}
    for s in range(2, len(model_names) + 1):
        comb_acc[s] = {}
        for comb in itertools.combinations(model_names, s):
            model_ids = "-".join([str(model_names.index(n)) for n in comb])
            if len(comb) < 2:
                raise RuntimeError("Ensemble requires at least 2 models")

            predictions, pred_arr = load_predictions(comb, n_query=n_query, n_way=n_way, set_name=mode)

            predictions["ensemble_majority"] = ensemble_methods["voting"](pred_arr, method="majority",
                                                                          n_way=n_way, n_query=n_query)
            predictions["ensemble_plurality"] = ensemble_methods["voting"](pred_arr, method="plurality",
                                                                           n_way=n_way, n_query=n_query)

            acc_dict = {}
            y = np.repeat(range(n_way), n_query)
            for model_n, pred in predictions.items():
                acc_score = []
                nan_per = np.isnan(pred).sum() / (n_way * n_query * n_sample) * 100
                # print(f"Nan percentage of {model_n}: {nan_per:.2f}")
                for i in range(n_sample):
                    nan_vals = np.isnan(pred[i])
                    if nan_vals.any():
                        if drop_nans:
                            acc = accuracy_score(y[~nan_vals], pred[i][~nan_vals]) * 100
                        else:
                            acc = np.mean(y == pred[i]) * 100
                    else:
                        acc = accuracy_score(y, pred[i]) * 100
                    acc_score.append(acc)
                acc_dict[model_n] = np.array(acc_score)
                if model_n == "ensemble_majority":
                    nan_comb[model_ids] = nan_per

            mjr = np.mean(acc_dict["ensemble_majority"])
            plr = np.mean(acc_dict["ensemble_plurality"])

            comb_acc[s][model_ids] = [mjr, plr]

            # for model_n, acc in acc_dict.items():
            #     acc_mean = np.mean(acc)
            #     acc_std = np.std(acc)
            #     ci = 1.96 * acc_std / np.sqrt(n_sample)
            #     acc_str = f"{model_n} {mode}: {acc_mean:.2f} +- {ci:.2f}"
        # print(acc_str)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for s, scores in comb_acc.items():
        x_axis = np.tile(s, len(scores.keys()))
        y_axis = np.array(list(comb_acc[s].values()))

        for i in range(2):
            ax[i].scatter(x_axis, y_axis[:, i], c="b")
            for k, (key, val) in enumerate(comb_acc[s].items()):
                ax[i].text(x_axis[k], val[i], key)
            ax[i].grid()
            ax[i].set_xlabel("Ensemble Size")
            ax[i].set_ylabel("Acc (%)")

    ax[0].set_title("Ensemble Majority")
    ax[1].set_title("Ensemble Plurality")

    best_model = np.tile(np.mean(acc_dict["DeepEMD"]), len(comb_acc.keys()))
    model_x_axis = np.array(list(comb_acc.keys()))
    for i in range(2):
        ax[i].plot(model_x_axis, best_model, "--", label="DeepEMD", lw=3, c="r")
        ax[i].legend()

    plt.savefig(f"figures/{mode}_ens_combinations_{drop_nans}.png", dpi=200, bbox_inches="tight")

    # # majority analysis
    # fig, ax = plt.subplots(figsize=(8, 6))
    # size_c = {s+1: f"C{s}" for s in range(1, len(model_names))}
    # split_by_size = {}
    # for comb, val in nan_comb.items():
    #     s = len(comb.split("-"))
    #     acc_val = comb_acc[s][comb][0]
    #     ax.text(val, acc_val, comb)
    #     if s not in split_by_size.keys():
    #         split_by_size[s] = [[val, acc_val]]
    #     else:
    #         split_by_size[s] += [[val, acc_val]]
    #
    # for s, arr in split_by_size.items():
    #     np_arr = np.array(arr)
    #     ax.scatter(np_arr[:, 0], np_arr[:, 1], c=size_c[s], label=s)
    # ax.grid(True)
    # ax.set_xlabel("Episode Drop (%)")
    # ax.set_ylabel("Acc (%)")
    # ax.set_title(f"Percentage of Episodes Dropped during Majority, Dataset={mode}", fontsize=10)
    # ax.legend()
    # plt.savefig(f"figures/{mode}_drop_per.png", dpi=200, bbox_inches="tight")

    print()

    # print("saving ensemble csv")
    # csv_dir = "csvs"
    # if not os.path.exists(csv_dir):
    #     os.makedirs(csv_dir)
    # ens_df = copy.deepcopy(predictions)
    # ens_df["Truth"] = np.tile(y, n_sample).reshape(n_sample, n_query * n_way)
    # for model_n, pred in ens_df.items():
    #     ens_df[model_n] = pred.flatten().astype(int)
    # ens_df = pd.DataFrame(ens_df)
    # ens_df.to_csv(os.path.join(csv_dir, f"{mode}_ens.csv"))

    # print("plotting ensemble")
    # if not os.path.exists("figures"):
    #     os.makedirs("figures")
    # x_axis = np.arange(n_sample)
    # fig, ax = plt.subplots(figsize=(12, 6))
    # for model_n, acc_score in acc_dict.items():
    #     if "ensemble" in model_n:
    #         ax.plot(x_axis, acc_score, lw=2, c=ens_colors[model_n], label=model_n)
    #     else:
    #         ax.plot(x_axis, acc_score, c=model_colors[model_n], alpha=0.3, label=model_n)
    #
    # x_axis = np.arange(n_sample)
    # for model_n, acc_score in acc_dict.items():
    #     if "ensemble" in model_n:
    #         ax.plot(x_axis, np.tile(acc_score.mean(), n_sample),
    #                 lw=3, c="k", label=f"{model_n} mean value", linestyle="--")
    #     else:
    #         ax.plot(x_axis, np.tile(acc_score.mean(), n_sample),
    #                 lw=3, c=model_colors[model_n], linestyle="--", label=f"{model_n} mean")
    #
    # ax.set_xticks(np.linspace(0, n_sample, 11))
    # ax.set_yticks(np.arange(0, 101, 10))
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.set_xlim(0, n_sample)
    # ax.set_title(f"{mode} accuracies in each episode", fontsize=15)
    # ax.set_xlabel("Number of Episodes", fontsize=15)
    # ax.set_ylabel("Accuracy (%)", fontsize=15)
    # ax.grid()
    # ax.legend()
    # plt.savefig(f"figures/{mode}_ens_{n_sample}.png", dpi=200, bbox_inches="tight")
    # plt.show()


if __name__ == '__main__':
    n_query = 15
    n_way = 5
    mode = "novel"
    n_sample = 1000
    drop_nans = False
    run()
