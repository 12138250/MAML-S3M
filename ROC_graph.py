import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

class ROC_graph(object):
    def __init__(self, logits, labels_test):
        self.logits = np.array(logits)
        self.labels_test = labels_test

    def roc(self):
        logits = self.logits
        labels_test = self.labels_test
        if hasattr(labels_test, 'cpu'):
            labels_test = labels_test.cpu().detach().numpy()
        labels_test = labels_test.flatten()
        cols = ['k', '#AE433D', '#F8DB51', 'skyblue', '#354E87', 'purple', 'green', 'orange', 'brown', 'pink']
        plt.figure(num='ROC Curve', figsize=(8, 6), facecolor='w', edgecolor='k', dpi=240)
        for j in range(10):
            y_score = logits[:, j]
            y_true_binary = (labels_test == j).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)

            if j == 0:
                plt.plot(fpr, tpr, '{}'.format(cols[j]),
                         label='Norm ({:.4f})'.format(roc_auc), lw=1.5)
            elif j == 1:
                plt.plot(fpr, tpr, '{}'.format(cols[j]),
                         label='IR-0.007 ({:.4f})'.format(roc_auc), lw=1.5, mew=15)
            elif j == 2:
                plt.plot(fpr, tpr, '{}'.format(cols[j]),
                         label='IR-0.014 ({:.4f})'.format(roc_auc), lw=1.5)
            elif j == 3:
                plt.plot(fpr, tpr, '{}'.format(cols[j]),
                         label='IR-0.021 ({:.4f})'.format(roc_auc), lw=1.5)
            elif j == 4:
                plt.plot(fpr, tpr, '{}'.format(cols[j]),
                         label='OR-0.007 ({:.4f})'.format(roc_auc), lw=1.5)
            elif j == 5:
                        plt.plot(fpr, tpr, '{}'.format(cols[j]),
                                 label='0R-0.014 ({:.4f})'.format(roc_auc), lw=1.5)
            elif j == 6:
                        plt.plot(fpr, tpr, '{}'.format(cols[j]),
                                 label='OR-0.021 ({:.4f})'.format(roc_auc), lw=1.5)
            elif j == 7:
                        plt.plot(fpr, tpr, '{}'.format(cols[j]),
                                 label='Ball-0.007 ({:.4f})'.format(roc_auc), lw=1.5)
            elif j == 8:
                        plt.plot(fpr, tpr, '{}'.format(cols[j]),
                                 label='Ball-0.014 ({:.4f})'.format(roc_auc), lw=1.5)
            else:
                        plt.plot(fpr, tpr, '{}'.format(cols[j]),
                                 label='Ball-0.021 ({:.4f})'.format(roc_auc), lw=1.5)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=24, labelpad=16)
        plt.ylabel('True Positive Rate', fontsize=24, labelpad=16)
        plt.legend(loc="lower right", fontsize=20)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/Proposed_roc.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.show()