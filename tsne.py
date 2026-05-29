import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.patches as mpatches


def plot_embedding(data, labels, classes=None):
    os.makedirs('./results', exist_ok=True)
    data_np = np.array(data)
    labels_np = np.array(labels).flatten()

    unique_labels = np.unique(labels_np)

    if classes is not None:
        if isinstance(classes, int):
            label_names = {i: str(i) for i in range(classes)}
            print(f"Auto-generated labels for {classes} classes: {list(label_names.values())}")
        elif isinstance(classes, list):
            label_names = {i: str(classes[i]) if i < len(classes) else str(i)
                           for i in unique_labels}
            print(f"Using provided class list: {classes}")
        elif isinstance(classes, dict):
            label_names = classes
            print(f"Using provided class dict: {classes}")
        else:
            label_names = {int(label): str(classes) for label in unique_labels}
            print(f"Using single class name: {classes}")
    else:
        label_names = {int(label): str(int(label)) for label in unique_labels}
        print(f"Using default labels: {label_names}")

    print(f"Performing t-SNE: {data_np.shape[1]}D -> 2D")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    data_2d = tsne.fit_transform(data_np)

    plt.figure(figsize=(12, 9))
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(len(unique_labels))]

    legend_handles = []
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        label_key = int(label)
        if label_key in label_names:
            label_name = label_names[label_key]
        else:
            label_name = f'Class {label_key}'

        plt.scatter(data_2d[mask, 0], data_2d[mask, 1],
                    c=[colors[i]], alpha=0.7, s=50,
                    edgecolors='w', linewidth=0.5,
                    label=label_name)
        legend_handles.append(mpatches.Patch(color=colors[i], label=label_name))


    plt.legend(handles=legend_handles,
               title=None,
               fontsize=20,
               loc='lower right',
               frameon=True,
               fancybox=True,
               shadow=True,
               borderpad=1)


    save_path = './results/real_tsne.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE figure saved to: {save_path}")
    plt.close()
    print(f"   Before: {data_np.shape}")
    print(f"   After: {data_2d.shape}")
    return plt.gcf()