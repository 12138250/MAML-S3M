import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import csv
import random

class Imagenet(Dataset):
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        self.imgsz = resize
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query
        self.resize = resize
        self.startidx = startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                            transforms.Resize((self.resize, self.resize)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                transforms.Resize((self.resize, self.resize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        self.path = root
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)
            self.img2label[k] = i + self.startidx
        self.cls_num = len(self.data)
        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        dictLabels = {}
        with open(csvf, encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        self.support_x_batch = []
        self.query_x_batch = []
        for b in range(batchsz):
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                cls_sample_count = len(self.data[cls])
                required_samples = self.k_shot + self.k_query
                if cls_sample_count < required_samples:
                    print(f"Warning: Class {cls} has only {cls_sample_count} samples, "
                          f"need {required_samples}. Using replacement.")
                    selected_imgs_idx = np.random.choice(cls_sample_count, required_samples, True)
                else:
                    selected_imgs_idx = np.random.choice(cls_sample_count, required_samples, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])
                support_x.append(np.array(self.data[cls])[indexDtrain].tolist())
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())
            random.shuffle(support_x)
            random.shuffle(query_x)
            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)

    def __getitem__(self, index):
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        support_y = np.zeros((self.setsz), dtype=int)
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        query_y = np.zeros((self.querysz), dtype=int)

        flatten_support_x = [os.path.join(self.path, item[:1], item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array( [self.img2label[item[:1]]
                    for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item[:1], item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:1]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        unique = np.unique(support_y)
        random.shuffle(unique)
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        for i, path in enumerate(flatten_support_x):
            path = path if os.path.exists(path) else path.replace('（', '(').replace('）', ')')
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            path = path if os.path.exists(path) else path.replace('（', '(').replace('）', ')')
            query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        return self.batchsz