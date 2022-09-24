#!/usr/bin/env python
# coding: utf-8

from main import ChessDataset, mAP
import numpy as np
import os
import torch
from engine import train_one_epoch
import torchvision as tv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import utils
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from warnings import simplefilter

simplefilter("ignore")

op = os.path
ROOT = "data/train/"


# In[2]:


ds = ChessDataset(ROOT)

model = tv.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 12

in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# In[3]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)


# In[4]:


# Do the stratification here
df = [
    (ann["id"].item(), ds.lenc.inverse_transform(ann["labels"].cpu())) for _, ann in ds
]
df = pd.DataFrame(df, columns=["id", "labels"])
df["count"] = df["labels"].apply(len)
df["labels"] = df["labels"].apply(lambda x: x[0] if len(x) == 1 else x)
singles = df[df["count"] == 1]
singles_train, singles_test = map(
    lambda x: x.index,
    train_test_split(singles, stratify=singles["labels"], train_size=0.6),
)
df.drop(singles_train, axis=0, inplace=True)
df.drop(singles_test, axis=0, inplace=True)
df["bin"] = pd.cut(df["count"], bins=10, labels=list("ABCDEFGHIJ"))

others_train, others_test = map(
    lambda x: x.index,
    train_test_split(df, stratify=df["bin"], train_size=0.6),
)

train_ix = np.r_[singles_train, others_train]
test_ix = np.r_[singles_test, others_test]

np.random.shuffle(train_ix)
np.random.shuffle(test_ix)

ds_train = torch.utils.data.Subset(ds, train_ix)
ds_test = torch.utils.data.Subset(ds, test_ix)

##

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=0.0005, momentum=0.9, weight_decay=1e-4, nesterov=True
)

data_loader_train = torch.utils.data.DataLoader(
    ds_train, batch_size=8, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
)
data_loader_test = torch.utils.data.DataLoader(
    ds_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
)
num_epochs = 20

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
for epoch in tqdm(range(num_epochs)):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=0)

torch.save(model, "data/models/strat.pt")


# In[189]:


_ = model.eval()

metrics = {}
for img, ann in tqdm(ds_test):
    pred = model([img.to(device)])[0]
    metrics[ann["id"].item()] = mAP(ann, pred, device)


metrics = pd.Series(metrics)
metrics.to_csv(f"data/output/chess_iou_strat_single_vs_more_{datetime.now().isoformat()}.csv")
print(metrics.mean())
