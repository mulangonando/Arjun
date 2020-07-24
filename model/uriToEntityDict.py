#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from requests import get
from bs4 import BeautifulSoup
pd.set_option('display.max_colwidth', -1)


# In[2]:


#df = pd.read_csv('../../dataset/entity_uri.csv', encoding='utf-8')
#df = df.drop_duplicates(subset=['Uri'])
#df.info()
#df.to_csv('../../dataset/entity_uri_dups.csv', encoding='utf-8', index=False)

#chunks = pd.read_csv('../../dataset/entity_uri.csv', encoding='utf-8', chunksize=10)
#df = pd.read_csv('../../dataset/entity_uri.csv',encoding='utf-8')
#df = pd.read_csv('../../dataset/entity_uri_dups.csv', usecols=['Entity','Uri'], encoding='utf-8')
#sum(df.duplicated(subset=['Uri']))


# In[3]:


chunks = pd.read_csv('../../dataset/entity_uri_dups.csv', usecols=['Entity','Uri'], encoding='utf-8', chunksize=1000)


# In[4]:


def extractEntityFromUri(url):
    try:
        #print url
        #print (url)
        url1 = 'https://www.wikidata.org/wiki/'+url.strip()
        #print (url1)
        response = get(url1, verify=True)
        #print url
        html_soup = BeautifulSoup(response.text, 'html.parser')
        data = html_soup.find('title')
        #URLObject = urllib2.urlopen(uri)
        #html = BeautifulSoup(URLObject.read())
        #data = html.find('title')
        #print data
    # Print title
        if 'Wikidata' in data.contents[0][-11:]:
            #print data.contents[0][:-11]
            return data.contents[0][:-11].strip()
        else:
            #print data.contents[0]
            return data.contents[0].strip()
    except:
        print ("error: parsing Error") 
        #print url,
    return 'EntityDonotExists'

def readChunkforQURI(chunk):
    chunk = chunk.dropna()
    seqList = chunk['uri'].tolist()
    '''for seq in seqList:
        #print seq.split()
        for url in set(seq.strip().split()):
            extractEntityFromUri(url)'''
            #print url
    #tmp_urlEntityDict = {url:extractEntityFromUri(url.strip()) for seq in seqList for url in set(seq.strip().split())}
    tmp_urlEntitySet = set([url.strip() for seq in seqList for url in set(seq.strip().split())])
    #tmp_urlEntityDict = [extractEntityFromUri(ur.strip()) for seq in seqList for url in set(seq.split()) for ur in url.split()]
    #print (tmp_urlEntityDict)
    return tmp_urlEntitySet

def extractEntityForChunk(chunk):
    chunk['UriEntity'] = chunk['Uri'].apply(extractEntityFromUri)
    return chunk


# In[5]:


#pools = mp.Pool(28)

#urlEntitySet = set()

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z
#for list of QURI from original dataSet
#for tmp_urlEntitySet in pools.map(readChunkforQURI, chunks):
#    #urlEntitySet = merge_two_dicts(urlEntityDict, tmp_urlEntityDict)
#    urlEntitySet = urlEntitySet.union(tmp_urlEntitySet)
#urlEntityDict = {uri:extractEntityFromUri(uri) for uri in urlEntitySet}
#print urlEntityDict
#print len(urlEntityDict)
#df = pd.DataFrame(urlEntityDict.items(), columns=['Entity','UriEntity','Uri(https://www.wikidata.org/wiki/)'])
#display(df.head(10))
#df.to_csv('../../dataset/dataentity_uri_entity.csv', encoding='utf-8', index=False)


df = pd.DataFrame()
#for chunk in pools.map(extractEntityForChunk, chunks):
#    df = df.append(chunk,ignore_index=True)
#pools.close()
#pools.join()

#print (df.head(10))

for i, chunk in enumerate(chunks):
    chunk = extractEntityForChunk(chunk)
    df = df.append(chunk,ignore_index=True)
    if i%5000==0:
        print(chunk.head(10))

#print (df.head(10))
df.to_csv('../../dataset/dataentity_uri_entity.csv', encoding='utf-8', index=False)


# In[6]:


#


# In[ ]:





# In[ ]:





# In[ ]:




