#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import multiprocessing as mp
pd.set_option('display.max_colwidth', -1)


# In[2]:


df = pd.read_csv('../../dataset/entityData/WikidataLabel_clean.csv', encoding='utf-8', header=None, names=['qValue', 'entity'])


# In[3]:


qDict = dict(zip(df.qValue, df.entity))

print (len(qDict))

del df

# In[17]:


def convertUriQvaluetoEntity(uris):
    #print (uris)
    strList = uris.strip().split()
    strList = [str(qDict[qValue]) for qValue in strList if qValue in qDict]
    uris = ' '.join(strList)
    return uris

def modifyChunk(chunk):
    chunk = chunk.dropna()
    chunk['uriSequence2'] = chunk.loc[:,'uri'].apply(convertUriQvaluetoEntity)
    return chunk
    
    


# In[18]:


chunks = pd.read_csv('../../dataset/entityData/merged_465_entity_uri.csv', encoding='utf-8', chunksize=10000)


# In[19]:

df = pd.DataFrame()

for chunk in chunks:
    chunk = chunk.dropna()
    chunk['uriSequence2'] = chunk['uri'].apply(convertUriQvaluetoEntity)
    df = df.append(chunk,ignore_index=True)

'''pools = mp.Pool(28)
for chunk in pools.map(modifyChunk, chunks):
    df = df.append(chunk,ignore_index=True)
pools.close()
pools.join()'''

print (df.info())
print (df.head(50))

df.to_csv('../../dataset/entityData/merged_465_entity_uri_entity.csv', encoding='utf-8', index=False)


# In[ ]:




