#!/usr/bin/env python
# coding: utf-8

import numpy as np
from main import COCODataset, mAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import os
import torch
from engine import train_one_epoch
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

ds = COCODataset("data/train", "data/annotations.json", intix=True)

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
sizes = []
for img, ann in ds:
    labels = ds.lenc.inverse_transform(ann["labels"])
    text = " ".join([k.replace(" ", "_") for k in labels])
    docs.append({"id": ann["id"].item(), "text": text})
    c, h, w = img.shape
    area = ann["area"] / (h * w)
    for label, p_area in zip(labels, area):
        sizes.append({"id": ann["id"].item(), "label": label, "area": p_area.item()})

# Create the sizes dataframe
sizes = pd.DataFrame.from_records(sizes)
tables_ix = sizes[sizes["label"] == "dining table"].index
sizes["dt_label"] = "none"
sizes.loc[tables_ix, "dt_label"] = np.where(
    sizes.loc[tables_ix]["area"] > 0.5, "large", "small"
)

sdf = pd.crosstab(index=sizes["id"], columns=sizes["dt_label"])
sdf["label"] = ""
sdf.loc[sdf[["large", "small"]].sum(axis=1) == 0, "label"] = "none"
dt_ix = sdf[sdf[["large", "small"]].sum(axis=1) > 0].index
sdf.loc[dt_ix, "label"] = sdf.loc[dt_ix][["large", "small"]].idxmax(axis=1)

cdf = pd.DataFrame.from_records(docs)
cdf.set_index("id", inplace=True, verify_integrity=True)
vect = CountVectorizer()
X = vect.fit_transform(cdf["text"].tolist())

km = KMeans().fit(X)
cdf["label"] = km.labels_
assert np.all(cdf.index == sdf.index)

cdf["label"] = cdf["label"].astype(str) + sdf["label"]

# Replace the classes appearing just once
lcv = cdf['label'].value_counts()
single = lcv[lcv == 1].index
cdf['label'].replace({k: 'single_val' for k in single}, inplace=True)

train_ix, test_ix = train_test_split(cdf, stratify=cdf["label"], train_size=120)

ds = COCODataset("data/train", "data/annotations.json", intix=False)
ds_train = torch.utils.data.Subset(ds, train_ix.index)
ds_test = torch.utils.data.Subset(ds, test_ix.index)

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

torch.save(model, "data/models/strat-cluster-tablesize.pt")
_ = model.eval()

metrics = {}
for img, ann in tqdm(ds_test):
    pred = model([img.to(device)])[0]
    metrics[ann["id"].item()] = mAP(ann, pred, device)


metrics = pd.Series(metrics)
metrics.to_csv(f"data/output/coco_cluster_tablesize{datetime.now().isoformat()}.csv")
print(metrics.mean())
