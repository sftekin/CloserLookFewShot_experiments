import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance

def run():
    model_names = ["DeepEMD", "protonet", "simpleshot"]
    model_feats = []
    for m_name in model_names:
        embed_path = f"features/embeddings/{m_name}/base/embed.pt"
        label_path = f"features/embeddings/{m_name}/base/label.pt"

        embed = torch.load(embed_path).squeeze().cpu().numpy()
        label = torch.load(label_path).squeeze().cpu().numpy()
        idx = np.random.randint(0, len(embed), 1000)
        pca = PCA(n_components=2)
        # pca_feats = pca.fit_transform(embed[idx])
        pca_feats = TSNE(n_components=2, learning_rate='auto', init='pca', metric=wasserstein_distance, perplexity=15).fit_transform(embed[idx])
        plt.figure()
        plt.scatter(pca_feats[:, 0], pca_feats[:, 1], c=label[idx])
        plt.title(m_name)
        plt.show()
        print()



if __name__ == '__main__':
    run()
