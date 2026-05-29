import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


class ConfusionMatrixPlotter:
    def __init__(self, pred_labels, true_labels, n_classes=None):
        self.pred_labels = np.array(pred_labels).flatten()
        self.true_labels = np.array(true_labels).flatten()
        if n_classes is None:
            n_classes = max(int(self.true_labels.max()), int(self.pred_labels.max())) + 1
        self.n_classes = n_classes
        assert len(self.pred_labels.shape) == 1, "pred_labels must be 1D"
        assert len(self.true_labels.shape) == 1, "true_labels must be 1D"
        assert len(self.pred_labels) == len(self.true_labels), "length mismatch"
        print(f"ConfusionMatrixPlotter initialized:")
        print(f"  Samples: {len(self.true_labels)}")
        print(f"  Classes: {self.n_classes}")

    def plot_confusion_matrix(self, save_dir='./results', normalize=True):
        os.makedirs(save_dir, exist_ok=True)
        cm = confusion_matrix(self.true_labels, self.pred_labels,
                                labels=range(self.n_classes))
        if normalize:
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            cm_display = cm_normalized
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
            vmin, vmax = 0, 1
        else:
            cm_display = cm
            title = 'Confusion Matrix'
            fmt = 'd'
            vmin, vmax = None, None

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                         xticklabels=range(self.n_classes),
                         yticklabels=range(self.n_classes),
                         vmin=vmin, vmax=vmax,
                         annot_kws={'size': 20},
                         cbar_kws={'label': 'Normalized Value' if normalize else 'Count'})
        plt.title(title, fontsize=18, pad=20)
        plt.ylabel('True Label', fontsize=30)
        plt.xlabel('Predicted Label', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)
        cbar.ax.yaxis.label.set_size(30)
        plt.tight_layout()

        filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved to: {save_path}")
        self._print_metrics(cm, cm_normalized if normalize else None)
        return cm, cm_normalized if normalize else cm

    def _print_metrics(self, cm, cm_normalized=None):
        print("\nClassification Metrics:")
        for i in range(self.n_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            precision = TP / (TP + FP + 1e-10)
            recall = TP / (TP + FN + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            print(f"Class {i}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

        accuracy = np.trace(cm) / cm.sum()
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if cm_normalized is not None:
            avg_class_acc = np.mean(np.diag(cm_normalized))
            print(f"Average Class Accuracy: {avg_class_acc:.4f} ({avg_class_acc*100:.2f}%)")