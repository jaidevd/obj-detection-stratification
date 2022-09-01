#!/usr/bin/env python
# coding: utf-8

# In[1]:


from main import (
    PennFudanDataset,
    get_instance_segmentation_model,
    get_transform,
    map_iou,
)
import pandas as pd
import torch
import utils
from engine import train_one_epoch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')

# In[2]:


# use our dataset and defined transformations
dataset = PennFudanDataset(
    "/media/jaidevd/motherbox/archive/pennfudan/PennFudanPed",
    None,
    get_transform(train=True),
)
dataset_test = PennFudanDataset(
    "/media/jaidevd/motherbox/archive/pennfudan/PennFudanPed",
    None,
    get_transform(train=False),
)

# Stratification: Nonparametric, by X and Y
df = []
for image, ann in dataset_test:
    masks = ann['masks'].numpy()
    n_ped, h, w = masks.shape
    fc = np.mean(masks.sum(-1).sum(-1) / (h * w))
    df.append({'id': ann['image_id'].item(), 'X': n_ped, 'Y': fc})
df = pd.DataFrame.from_records(df)
edges = np.round(np.arange(0.02, 0.22, 0.02), 2)
df['ylabel'] = pd.cut(df['Y'], bins=10, labels=edges)
bins = df[['X', 'ylabel']].drop_duplicates()
bins['label'] = np.arange(bins.shape[0])
bins.set_index(['X', 'ylabel'], inplace=True, verify_integrity=True)
df['bin_id'] = df.apply(lambda x: bins.loc[(x['X'], x['ylabel'])], axis=1)
vc = df['bin_id'].value_counts()
single_bin = vc[vc == 1]

xdf = df[~df['bin_id'].isin(single_bin.index)]
train, test = train_test_split(xdf, stratify=xdf['bin_id'], train_size=100)
test = pd.concat((test, df[df['bin_id'].isin(single_bin.index)]))

# End stratification


# split the dataset in train and test set
# torch.manual_seed(1)
# indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, train.index)
dataset_test = torch.utils.data.Subset(dataset_test, test.index)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=5, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn,
)


# In[3]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# In[4]:


# let's train it for 10 epochs

num_epochs = 20

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    losses = train_one_epoch(
        model, optimizer, data_loader, device, epoch, print_freq=0
    )
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)

metrics = {}
model.eval()

for ((image,), (ann,)) in tqdm(data_loader_test):
    imid = ann["image_id"].item()
    y_pred = model([image.to("cuda")])[0]

    yt = ann["masks"].to("cpu").numpy()
    yp = y_pred["masks"].to("cpu").detach().numpy()
    yp = np.squeeze(yp, axis=1)
    metrics[imid] = map_iou(yt.astype(bool), yp > 0.5), yt.astype(bool), yp > 0.5

ious = pd.Series({k: v[0] for k, v in metrics.items()})
print(ious.mean())
