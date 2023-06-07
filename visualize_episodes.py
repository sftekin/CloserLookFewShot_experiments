import os
import pickle as pkl
import torch
import configs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from data_manager.episode_loader import EpisodeSet, TransformLoader
import matplotlib.gridspec as gridspec
import seaborn as sns


def get_sample_indexes(dataset, batch_sampler, n_query, n_way):
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

    all_sample_idx = torch.stack(my_batch)
    support_idx = all_sample_idx[:, :, 0]
    query_idx = all_sample_idx[:, :, 1:].reshape(-1, n_query*n_way)

    return support_idx, query_idx


def visualize_episode(input_idx, support_idx, query_idx, dataset, batch_sampler, n_query, n_way, save_name):

    episode_idx = input_idx // (n_query * n_way)
    support_indexes = support_idx[episode_idx]
    query_idx = query_idx[episode_idx, input_idx % (n_query * n_way)]

    all_classes = list(dataset.labels)
    support_classes = [all_classes[i] for i in batch_sampler[episode_idx]]
    label_idx = np.repeat(range(n_way), n_query)[input_idx % (n_query * n_way)]
    query_class = support_classes[label_idx]

    support_paths = []
    for sup_cls, sup_idx in zip(support_classes, support_indexes):
        support_paths.append(dataset.class2files[sup_cls][sup_idx])
    query_path = dataset.class2files[query_class][query_idx]

    model_names = stats["model_names"]
    base_logits = stats["base_logits"][input_idx].reshape((len(model_names), n_way))
    base_preds = base_logits.argmax(axis=1)
    ens_logits = stats["ens_logits"][input_idx]
    ens_pred = ens_logits.argmax()

    plt.style.use("seaborn")
    fig = plt.figure(tight_layout=True, figsize=(10, 5))
    gs = gridspec.GridSpec(2, 6)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(Image.open(query_path).resize((200, 200), Image.Resampling.LANCZOS))
    ax.axis("off")
    ax.set_title("Query")
    for i in range(len(support_paths)):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(Image.open(support_paths[i]).resize((200, 200), Image.Resampling.LANCZOS))
        if i == label_idx:
            ax.set_title(f"target")
        ax.set_yticks([])
        ax.set_xticks([100])
        x_tick_label = f"Support-{i}"
        for model_n, pred_idx in zip(model_names, base_preds):
            if i == pred_idx:
                x_tick_label += f"\n{model_n.split('_')[0]}"
        if i == ens_pred:
            x_tick_label += "\nEnsemble"
        ax.set_xticklabels([x_tick_label])

    x_axis = np.arange(n_way)
    ax = fig.add_subplot(gs[1, 1:])
    model_names_ = ["Ensemble"] + model_names
    base_logits_ = np.concatenate([ens_logits[None, :], base_logits])
    for i in range(len(model_names_)):
        sns.histplot(x=x_axis, weights=base_logits_[i], kde=True, discrete=True, kde_kws={'cut': 1},
                     line_kws={'linewidth': 4}, label=model_names_[i].split("_")[0], ax=ax)
        x_axis = x_axis + n_way
        ax.legend()
    ax.set_xticks([])
    ax.set_ylabel("Logits", fontsize=16)
    save_path = f"figures/{save_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}/{input_idx}.png", bbox_inches="tight", dpi=200)
    # plt.show()


if __name__ == '__main__':

    dataset_name = "miniImagenet"
    n_query = 15
    n_way = 5
    n_shot = 1
    n_episodes = 600
    image_size = 84

    base_file = configs.data_dir[dataset_name] + 'novel.json'

    transform = TransformLoader(image_size=image_size,
                                augmentation=False)
    dataset = EpisodeSet(meta_file_path=base_file,
                         transform=transform.get_transform(),
                         batch_size=n_shot + n_query,
                         max_batch_count=n_episodes,
                         load_sampler_indexes=True)

    batch_sampler_path = os.path.join(dataset.sampler_dir, f"{dataset.dataset}_{n_episodes}_batch_sampler.pkl")

    with open(batch_sampler_path, "rb") as f:
        batch_sampler = pkl.load(f)

    with open("inference_stats.pkl", "rb") as f:
        stats = pkl.load(f)

    support_idx, query_idx = get_sample_indexes(dataset, batch_sampler, n_query, n_way)

    selected_arr = stats["ens_stats"]["protonet_ResNet18-simpleshot_ResNet18"].squeeze()

    rand_int = np.random.permutation(len(selected_arr))[:100]
    count = 0
    for j in selected_arr[rand_int]:
        print(f"{count}/{len(rand_int)}")
        visualize_episode(j, support_idx, query_idx, dataset,
                          batch_sampler, n_query, n_way, save_name="proto_simpleshot_ensemble")
        count += 1

