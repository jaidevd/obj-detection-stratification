#!/usr/bin/env python
# coding: utf-8

# In[1]:


from main import COCODataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


# In[3]:


ds = COCODataset('data/train', 'data/annotations.json')

docs = []
for _, ann in ds:
    labels = ds.lenc.inverse_transform(ann['labels'])
    text = ' '.join([k.replace(' ', '_') for k in labels])
    docs.append({'id': ann['id'].item(), 'text': text})

df = pd.DataFrame.from_records(docs)
df.head()


# In[4]:


vect = CountVectorizer()
X = vect.fit_transform(df['text'].tolist())


# In[6]:


km = KMeans().fit(X)
df['label'] = km.labels_


# In[7]:


df.head()


# In[11]:


df[['id', 'label']].to_csv('data/clusters.csv', index=False)

