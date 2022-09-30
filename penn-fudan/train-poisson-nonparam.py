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
from warnings import simplefilter
simplefilter('ignore')

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

# Stratification: Nonparametric, by number of objects per image
X = []
for image, ann in dataset_test:
    X.append({"id": ann["image_id"].item(), "X": ann["masks"].shape[0]})
df = pd.DataFrame.from_records(X)
df.set_index("id", inplace=True, verify_integrity=True)
df.loc[df['X'] == 8, 'X'] = 7


train, test = train_test_split(df, stratify=df["X"], train_size=100)
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

# In[5]:


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
