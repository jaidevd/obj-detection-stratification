#!/usr/bin/env python
# coding: utf-8

from main import COCODataset, mAP, train_one_epoch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import os
import torch
import torchvision as tv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from warnings import simplefilter

simplefilter("ignore")

op = os.path

ds = COCODataset("data/train", "data/annotations.json")

model = tv.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 12

in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# In[3]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# Begin stratification
docs = []
for _, ann in ds:
    labels = ds.lenc.inverse_transform(ann["labels"])
    text = " ".join([k.replace(" ", "_") for k in labels])
    docs.append({"id": ann["id"].item(), "text": text})

cdf = pd.DataFrame.from_records(docs)
vect = CountVectorizer()
X = vect.fit_transform(cdf["text"].tolist())


# In[6]:


km = KMeans().fit(X)
cdf["label"] = km.labels_

train_ix, test_ix = train_test_split(cdf["id"], stratify=cdf["label"], train_size=120)

ds = COCODataset("data/train", "data/annotations.json", intix=True)
ds_train = torch.utils.data.Subset(ds, train_ix.values)
ds_test = torch.utils.data.Subset(ds, test_ix.values)

# Finish ####

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=0.0005, momentum=0.9, weight_decay=1e-4, nesterov=True
)

data_loader_train = torch.utils.data.DataLoader(
    ds_train, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
)
data_loader_test = torch.utils.data.DataLoader(
    ds_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
)
num_epochs = 20

for epoch in tqdm(range(num_epochs)):
    # train for one epoch, printing every 10 iterations
    loss = train_one_epoch(
        model, optimizer, data_loader_train, device, epoch, print_freq=0
    )

torch.save(model, "data/models/strat.pt")
_ = model.eval()

metrics = {}
for img, ann in tqdm(ds_test):
    pred = model([img.to(device)])[0]
    metrics[ann["id"].item()] = mAP(ann, pred, device)


metrics = pd.Series(metrics)
metrics.to_csv(f"data/output/coco_cluster_{datetime.now().isoformat()}.csv")
print(metrics.mean())
