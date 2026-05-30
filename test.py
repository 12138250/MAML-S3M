import torch, os
import numpy as np
from ImageNet import Imagenet
import scipy.stats
from torch.utils.data import DataLoader
import argparse
from Seed import set_seed
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from learner import Learner
from torch import nn, optim
from torch.nn import functional as F
from copy import deepcopy

from confusion_matrix import ConfusionMatrixPlotter
from ROC_graph import ROC_graph
from tsne import plot_embedding

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'


class Meta(nn.Module):
    def __init__(self, args, config):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.update_beita_lr = args.update_beita_lr

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim_base = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters(),
                                       retain_graph=True, allow_unused=True)

            grad = [g if g is not None else torch.zeros_like(p)
                    for g, p in zip(grad, self.net.parameters())]

            fast_weights = list(
                map(lambda p: (1 - self.update_lr * self.update_beita_lr) * p[1] - self.update_lr * p[0],
                    zip(grad, self.net.parameters()))
            )

            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
     

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()


            for k in range(1, self.update_step):
      

                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                grad = [g if g is not None else torch.zeros_like(p)
                        for g, p in zip(grad, fast_weights)]

                fast_weights = list(
                    map(lambda p: (1 - self.update_lr * self.update_beita_lr) * p[1] - self.update_lr * p[0],
                        zip(grad, fast_weights))
                )

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        loss_q = losses_q[-1] / task_num
        self.meta_optim_base.zero_grad()
        loss_q.backward()


        accs = np.array(corrects) / (querysz * task_num)
        return accs

    def testing(self, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)

        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters(), allow_unused=True)
        grad = [g if g is not None else torch.zeros_like(p)
                for g, p in zip(grad, net.parameters())]

        fast_weights = list(
            map(lambda p: (1 - self.update_lr * self.update_beita_lr) * p[1] - self.update_lr * p[0],
                zip(grad, net.parameters()))
        )

        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)

            grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
            grad = [g if g is not None else torch.zeros_like(p)
                    for g, p in zip(grad, fast_weights)]

            fast_weights = list(
                map(lambda p: (1 - self.update_lr * self.update_beita_lr) * p[1] - self.update_lr * p[0],
                    zip(grad, fast_weights))
            )

            if k >= 0 and k <= self.update_step_test - 2:
                corrects[k + 1] = 0

            if k == self.update_step_test - 1:
                logits_q = net(x_qry, fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        del net
        acc = np.array(corrects) / querysz
        return acc, pred_q, y_qry, logits_q


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, default=100, help='Training epochs')
    argparser.add_argument('--n_way', type=int, default=10, help='N-way classification')
    argparser.add_argument('--k_spt', type=int, default=5, help='Support set samples per class')
    argparser.add_argument('--k_qry', type=int, default=5, help='Query set samples per class')
    argparser.add_argument('--imgsz', type=int, default=120, help='Image size')
    argparser.add_argument('--imgc', type=int, default=3, help='Image channels')
    argparser.add_argument('--task_num', type=int, default=2, help='Number of tasks (meta batch size)')
    argparser.add_argument('--meta_lr', type=float, default=1e-3, help='Meta learning rate')
    argparser.add_argument('--update_lr', type=float, default=0.02, help='Inner loop learning rate')
    argparser.add_argument('--update_step', type=int, default=5, help='Inner loop update steps')
    argparser.add_argument('--update_step_test', type=int, default=10, help='Fine-tuning steps at test')
    argparser.add_argument('--update_beita_lr', type=int, default=1, help='Weight guidance factor (WGF)')

    # Data paths
    argparser.add_argument('--source_path', type=str,
                           default='CWRU_images/Load_0HP/images', help='Source domain data path')
    argparser.add_argument('--target_path', type=str,
                           default='CWRU_images/Load_2HP/images', help='Target domain data path')
    argparser.add_argument('--source_mode', type=str, default='0HP', help='Source mode')
    argparser.add_argument('--target_mode', type=str, default='2HP', help='Target mode')

    # Class names for visualization
    argparser.add_argument('--class_names', type=str, nargs='+',
                           default=None, help='Class name list')

    args = argparser.parse_args()

    set_seed(123)

    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Image size: {args.imgsz}x{args.imgsz}")
    print(f"Task: {args.n_way}-way, {args.k_spt}-shot, {args.k_qry}-query")
    print(f"Source -> Target: {args.source_mode} -> {args.target_mode}")
    print("=" * 80 + "\n")

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('se', []),

        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('csssm', [32, 2, 16, 2.0, 4.0, False]),

        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('csssm', [32, 2, 16, 2.0, 4.0, False]),

        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('se', []),

        ('flatten', []),
        ('linear', [args.n_way, 32 * 10 * 10])
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Meta(args, config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nLoading data...")
    Source_data = Imagenet(
        args.source_path,
        mode=args.source_mode,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        batchsz=3,
        resize=args.imgsz
    )

    Target_data = Imagenet(
        args.target_path,
        mode=args.target_mode,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        batchsz=10,
        resize=args.imgsz
    )

    print(f"Source: {args.source_mode}, Target: {args.target_mode}")

    if args.class_names is None:
        class_names = [f'Class_{i}' for i in range(args.n_way)]
    else:
        class_names = args.class_names

    best_acc = 0.0
    best_epoch = 0
    acc_history = []


    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(args.epoch):
        db = DataLoader(Source_data, args.task_num, shuffle=True,
                        num_workers=0, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = (
                x_spt.to(device), y_spt.to(device),
                x_qry.to(device), y_qry.to(device)
            )

            accs = model(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                print(f'Epoch {epoch:03d} | Step {step:04d} | Train Acc: {accs[-1]:.4f}')

            if step % 500 == 0:
                print(f"\n{'=' * 70}")
                print(f"Testing - Epoch {epoch:03d}, Step {step:04d}")
                print(f"{'=' * 70}")

                db_test = DataLoader(Target_data, 1, shuffle=True,
                                     num_workers=0, pin_memory=True)
                accs_all_test = []

                pred_labels = []
                true_labels = []
                logits_list = []
                softmax_features = []

                for x_spt_test, y_spt_test, x_qry_test, y_qry_test in db_test:
                    x_spt_test, y_spt_test, x_qry_test, y_qry_test = (
                        x_spt_test.squeeze(0).to(device),
                        y_spt_test.squeeze(0).to(device),
                        x_qry_test.squeeze(0).to(device),
                        y_qry_test.squeeze(0).to(device)
                    )

                    acc_single, pred_single, true_single, logit_single = model.testing(
                        x_spt_test, y_spt_test, x_qry_test, y_qry_test
                    )
                    accs_all_test.append(acc_single)

                    pred_labels.append(pred_single)
                    true_labels.append(true_single)
                    logits_list.append(logit_single)
                    softmax_prob = torch.nn.Softmax(dim=1)(logit_single)
                    softmax_features.append(softmax_prob)

                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                accs_std = np.array(accs_all_test).std(axis=0).astype(np.float16)

                if accs[-1] > best_acc:
                    best_acc = accs[-1]
                    best_epoch = epoch



                    os.makedirs('./results', exist_ok=True)
                    torch.save(model.state_dict(),
                               f'./results/best.pth')
                    print(f"Saved new best model (Acc: {best_acc * 100:.2f}%)")

                acc_history.append(accs[-1])

                print(f"Test Acc: {accs[-1] * 100:.2f}% ± {accs_std[-1] * 100:.2f}%")
                print(f"Best Acc: {best_acc * 100:.2f}% (Epoch {best_epoch})")
                print(f"{'=' * 70}\n")




if __name__ == '__main__':
    main()
