import os
import pickle as pkl
import torch
import configs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from data_manager.episode_loader import EpisodeSet, TransformLoader


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


def visualize_episode(input_idx, dataset, batch_sampler, n_query, n_way):
    support_idx, query_idx = get_sample_indexes(dataset, batch_sampler, n_query, n_way)

    episode_idx = input_idx // (n_query * n_way)
    support_indexes = support_idx[episode_idx]
    query_idx = query_idx[episode_idx, input_idx % (n_query * n_way)]

    all_classes = list(dataset.labels)
    support_classes = [all_classes[i] for i in batch_sampler[episode_idx]]
    query_class = support_classes[input_idx % n_way]

    support_paths = []
    for sup_cls, sup_idx in zip(support_classes, support_indexes):
        support_paths.append(dataset.class2files[sup_cls][sup_idx])

    query_path = dataset.class2files[query_class][query_idx]

    fig, ax = plt.subplots()
    img = mpimg.imread('your_image.png')

    fig, ax = plt.subplots(1, 6, figsize=(10, 5))
    ax[0].imshow(Image.open(query_path).resize((200, 200), Image.Resampling.LANCZOS))
    ax[0].axis("off")
    ax[0].set_title("Query")
    for i in range(len(support_paths)):
        ax[i + 1].imshow(Image.open(support_paths[i]).resize((200, 200), Image.Resampling.LANCZOS))

        ax[i + 1].set_title(f"Support-{i}")
        ax[i + 1].set_yticks([])
        ax[i + 1].set_xticks([100])
        ax[i + 1].set_xticklabels(["ASDSAD\nkdalsakdlas"])
    plt.show()

    # img = Image.open(self.filenames[item]).convert('RGB')


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

    input_idx = stats["ens_stats"]["ensemble"].squeeze()[0]

    visualize_episode(input_idx, dataset, batch_sampler, n_query, n_way)


