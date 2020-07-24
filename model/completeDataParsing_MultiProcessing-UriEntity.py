#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from IPython.display import Markdown, display
from unicodedata import normalize
import string
import re
#pd.set_option('display.max_colwidth', -1)
import tensorflow as tf
nltk.download('punkt')
import numpy as np
import os
#import urllib2
#from bs4 import BeautifulSoup
#from requests import get
import multiprocessing as mp
import time


# In[2]:


print("Hello To CompleteParser")
df_file = pd.DataFrame()
df_file_entityUri = pd.DataFrame()
#urlEntityDict = dict()
urlEntityDict = []
path = '../../dataset/Trex/'
doc_id = 0

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

# In[ ]:


def searchEntityinSentence(entityList, sentence):
    global n_entity_in_sent
    global count_sentence
    #words_list = word_tokenize(sentence)
    #sequence2 = list(set([value for value in words_list if value in entityList])) remove duplicates
    sequence2 = [entity for entity in entityList if entity in sentence]
    sentence2 = ' '.join(word for word in sequence2)
    return sentence2

def extractEntityFromUri(url):
    try:
        url = 'https://www.wikidata.org/wiki/'+url
        response = get(url, verify=True)
        #print url
        html_soup = BeautifulSoup(response.text, 'html.parser')
        data = html_soup.find('title')
        #URLObject = urllib2.urlopen(uri)
        #html = BeautifulSoup(URLObject.read())
        #data = html.find('title')
        #print data
    # Print title
        if 'Wikidata' in data.contents[0][-11:]:
            return data.contents[0][:-11]
        else:
            return data.contents[0]
    except:
        return ''
    
    
def parseDoc(docid, sentences, entityJson, sentences_boundaries):
    global df_file
    global doc_id
    #sentence_list = sent_tokenize(sentences)
    entity_list_dict = {entity['boundaries'][0]:entity['surfaceform'] for entity in entityJson 
                       if  entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker' 
                        #or entity['annhttps://hadyelsahar.github.io/t-rex/samples/sample-output.jsonotator'] == 'Wikentity_list_dict = {entity['boundaries'][0]:entity['surfaceform'] for entity in entityJson 
                       #if  entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker' 
                        #or entity['annotator'] == 'Wikidata_Property_Linker'
                        }
    entity_list_dict_sep = {entity['boundaries'][0]:entity['surfaceform']+' SURFACEFORMSEPARATION' for entity in entityJson 
                       if  entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker' 
                        }
    entity_list_dict_uri = {entity['boundaries'][0]:entity['uri'] for entity in entityJson 
                       if  entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker' 
                        #or entity['annotator'] == 'Wikidata_Property_Linker'
                        }
    entity_list = [entity_list_dict[key].strip() for key in sorted(entity_list_dict.keys())]
    entities = ' '.join(entity for entity in entity_list)
    
    entity_list_sep = [entity_list_dict_sep[key].strip() for key in sorted(entity_list_dict_sep.keys())]
    entities_sep = ' '.join(entity for entity in entity_list_sep)
    
    entity_list_uri = [entity_list_dict_uri[key].replace('http://www.wikidata.org/entity/', '').strip() for key in sorted(entity_list_dict_uri.keys())]
    entities_uri = ' '.join(entity for entity in entity_list_uri)

    df = pd.DataFrame()
    if len(entity_list)==len(entity_list_uri):
        entity_uriEntity = [(entity.strip(),entityUri.strip()) for entity, entityUri in zip(entity_list, entity_list_uri)]
        #print entity_uriEntity
        #d = {'docid':docid,'sequence1':sentences, 'sequence2':entities, 'ui':entities_uri, 'entity:uriEntity':entity_uriEntity}
        d = {'docid':docid.replace('http://www.wikidata.org/entity/', '').strip(),'sequence1':sentences, 'sequence2':entities, 'sequence2Sep':entities_sep, 'uri':entities_uri}
        df = pd.DataFrame(data=d, index=[0])
        #df = pd.DataFrame(data=d)
        df = df.fillna('')
        #print(df)
    else:
        pass
    #doc_id = doc_id + 1
    return df,entity_uriEntity
    #return df
    

def paserFile(filename):
    df = pd.read_json(filename,encoding='utf-8')
    df = df[[u'docid',u'text',u'entities', u'sentences_boundaries']]
    df_sfile = pd.DataFrame()
    #file_entity_uriEntity = dict()
    file_entity_uriEntity = []
    for i in range(len(df.index)):
        df_d, entity_uriEntity = parseDoc(df.iloc[i,0], df.iloc[i,1], df.iloc[i,2],df.iloc[i,3])
        #file_entity_uriEntity = merge_two_dicts(file_entity_uriEntity, entity_uriEntity)
        file_entity_uriEntity = file_entity_uriEntity + entity_uriEntity
        df_sfile = df_sfile.append(df_d,ignore_index=True)
    return df_sfile,file_entity_uriEntity
    #return df_sfile
     
        
        


# In[ ]:


filesList = sorted(os.listdir(path))
pool = mp.Pool(mp.cpu_count())

files_20 = [path+file for file in filesList]
#print (files_20)

#for file in files_20:
#    df_sfile,tmp_urlEntityDict  = paserFile(file)
#    df_file = df_file.append(df_sfile,ignore_index=True)
#    urlEntityDict = merge_two_dicts(urlEntityDict, tmp_urlEntityDict)

    
for result in pool.map(paserFile, files_20):
    startTime = time.time()
    df_file = df_file.append(result[0],ignore_index=True)
    elapsedTime = time.time() - startTime
    #urlEntityDict = merge_two_dicts(urlEntityDict, result[1])
    urlEntityDict = urlEntityDict + result[1]
    print ("TimeTakenByFileParsing", elapsedTime)
    del result
    #print (df_file.info())
    
pool.close()
pool.join()

path = '../../dataset/entityData_Sep/'

df_file = df_file.dropna()
df_file = df_file.drop_duplicates(keep='first')
df_file = df_file.reset_index(drop=True)
df_file.to_csv(path+'merged_465_entity_uri_sep'+'.csv',encoding='utf-8',index=False)
print (df_file.info())
#display(df_file.head(50))

#df_entity_uri = pd.DataFrame(urlEntityDict.items(), columns=['Surface-Form', 'QUri'])
df_entity_uri = pd.DataFrame.from_records(urlEntityDict, columns=['Surface-Form', 'QUri'])
df_entity_uri = df_entity_uri.dropna()
df_entity_uri = df_entity_uri.drop_duplicates(keep='first')
df_entity_uri = df_entity_uri.reset_index(drop=True)
df_entity_uri.to_csv(path+'surfaceForm_wikiUri_sep'+'.csv',encoding='utf-8',index=False)
print (df_file.info())
#display(df_entity_uri.head(50))

del urlEntityDict
del df_file
del df_entity_uri



# In[3]:


#replace WikiUri with WikiDataEntity
df = pd.read_csv('../../dataset/entityData/WikidataLabel_clean.csv', encoding='utf-8', header=None, names=['qValue', 'entity'])
qDict = dict(zip(df.qValue, df.entity))
def convertUriQvaluetoEntity(uris):
    #print (uris)
    strList = uris.strip().split()
    strList = [str(qDict[qValue])+' WIKIDATAENTITYSEPARATION' for qValue in strList if qValue in qDict]
    uris = ' '.join(strList)
    return uris

def modifyChunk(chunk):
    chunk = chunk.dropna()
    chunk.loc['uriSequence2'] = chunk.loc[:,'uri'].apply(convertUriQvaluetoEntity)
    return chunk
    
chunks = pd.read_csv('../../dataset/entityData_Sep/merged_465_entity_uri_sep.csv', encoding='utf-8', chunksize=10000)

df = pd.DataFrame()
for chunk in chunks:
    chunk = chunk.dropna()
    chunk['uriSequence2'] = chunk['uri'].apply(convertUriQvaluetoEntity)
    df = df.append(chunk,ignore_index=True)

'''pools = mp.Pool(28)
for chunk in pools.map(modifyChunk, chunks):
    df = df.append(chunk,ignore_index=True)
    break
pools.close()
pools.join()'''

print (df.info())
#display (df.head(50))

df.to_csv('../../dataset/entityData_Sep/merged_465_entity_uri_entity_sep.csv', encoding='utf-8', index=False)

print (df.info())
#display(df.head(50))
del df
del qDict
del chunks


# In[9]:


#clean data according to google  or glove

def normaliseUnicodeCharacter(text):
    text = normalize('NFD', text).encode('ascii', 'ignore')
    text = text.decode('UTF-8')
    return text

def textToRegexpWordList(text):
    #text_tokens = nltk.word_tokenize(text)
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(text)
    text = ' '.join(text_tokens)
    return text

def textToList(text):
    text_tokens = nltk.word_tokenize(text)
    return text_tokens

def convertYeartoDatetoken(text_tokens):
    text_tokens = [word.lower() for word in text_tokens]
    return text_tokens

def removeNonAsciiWords(text_tokens):
    text_tokens = [word for word in text_tokens if all(ord(char) < 128 for char in word)==True]
    return text_tokens
       
def toLowercaseWordsList(text_tokens):
    text_tokens = [word.lower() for word in text_tokens]
    return text_tokens

def removePunctuation(text_tokens):
    #table = string.maketrans(", ",string.punctuation) #to change
    #text_tokens = [word.translate(table) for word in text_tokens]
    text_tokens = re.sub(r'[^\w\s]','',text_tokens)
    return text_tokens
    
def removeNonPrintablechars(text_tokens):
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    text_tokens = [re_print.sub('', w) for w in text_tokens]
    return text_tokens
    
def removeTokenwithNumbers(text_tokens):
    text_tokens = [word for word in text_tokens if word.isalpha()]
    return text_tokens

def removeGoogleStopWords(text):
    text_tokens = text.split()
    text_tokens = [word for word in text_tokens if word not in stopWords_google]
    text = ' '.join(text_tokens)
    return text

def textToWordTokenTotext(text):
    text_tokenss = nltk.word_tokenize(text)
    text = ' '.join(text_tokenss)
    return text

def nonVectorizedGoogleWords(text):
    words = []
    for word in text.split(): 
        try:
            m_google = model_google[word]
            #print word
        except:
             words.append(word)
    return words

def nonVectorizedGloveWords(text_tokenss):
    words = []
    for word in text_tokenss: 
        try:
            m_glove = model_glove[word]
            #print word
        except:
             words.append(word)
    return words

def textToWordSeqGoogle(text):
    text_tokenss = tf.keras.preprocessing.text.text_to_word_sequence(text, 
    filters='-!\'"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, 
    split=' ')
    #display(' '.join(text_tokenss))
    text = ' '.join(text_tokenss)
    return text

def textToWordSeqGlove(text):
    text_tokenss = tf.keras.preprocessing.text.text_to_word_sequence(
    text,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ')
    #display(' '.join(text_tokenss))
    return text_tokenss

patternNum = re.compile(u'(?:[1-2][0-9]{3}\u2013[0-9]{2}[0-9]?[0-9]?$|[1-2][0-9]{3}$)')

years = [u'1','1987', u'1935\u20131936', '1987-88', '88-89', 'AB-1987', '1987-CD', 'CD1987', '2003-', '1987AB', '223', 
         '3000' ,'12', 'abc1','ab12cd','0a' , 'abc']
numbers= ['012345', '101', '1010.2','as' , '123$', '0123.5']

#atleast one letter and one number
def isAlphaNumbers(word):
    letter_flag = False
    number_flag = False
    for c in word:
        if c.isalpha():
            letter_flag = True
        if c.isdigit():
            number_flag = True
    return letter_flag and number_flag


def wordTOYear(word):
    if re.match(patternNum, word):
        return '<YEAR>'
    else:
        return word

def numToNumber(word):
    wordNew = word 
    try:
        wordNew = float(wordNew)
        if wordNew > 31:
            return '<NUMBER>'
        else:
            return word
    except:
        return word

def replaceYears(text_tokens):
    #print ("replaceYear1: ",text_tokens)
    text_tokens = [wordTOYear(word) for word in text_tokens]
    #print("replaceYear: ", text_tokens)
    return text_tokens


def replaceNumbers(text_tokens):
    text_tokens = [numToNumber(word) for word in text_tokens]
    return text_tokens

def removeAlphaNum(text_tokens):
    text_tokens = [word for word in text_tokens if isAlphaNumbers(word)==False]
    return text_tokens
    
    
    
def textTotoken(text):
    return text.split()

def tokenTotext(text_tokens):
    return ' '.join(text_tokens)


def applyFunSeries(text):
    text = textToWordTokenTotext(text)
    text = textToWordSeqGoogle(text)
    text_token = textTotoken(text)
    text_token = replaceYears(text_token)
    text_token = replaceNumbers(text_token)
    text_token = removeAlphaNum(text_token)
    text = tokenTotext(text_token)
    return text



def textlen(text):
    text_token = text.split()
    return len(text_token)

def chunkTaskWorker(chunk):
    chunk = chunk.fillna('')
    #chunk.loc[:,'sequence1':'uriSequence2'] = chunk.loc[:,'sequence1':'uriSequence2'].applymap(applyFunSeries)
    chunk.loc[:,['sequence1', 'sequence2', 'sequence2Sep','uriSequence2']] = chunk.loc[:,['sequence1', 'sequence2', 'sequence2Sep','uriSequence2']].applymap(applyFunSeries)
    return chunk

n_pool = mp.cpu_count()
pool = mp.Pool(28)

path = '../../dataset/entityData_Sep/'
df_file = pd.DataFrame()
chunks = pd.read_csv(path+'merged_465_entity_uri_entity_sep.csv', dtype= str, encoding='utf-8',usecols=['docid','uri','sequence1', 'sequence2', 'sequence2Sep','uriSequence2' ], chunksize=10000)
print (n_pool,n_pool)

df_file = pd.DataFrame()

for chunk in pool.imap(chunkTaskWorker,chunks, chunksize=1):
    df_file = df_file.append(chunk,ignore_index=True)

#display(df_file.head(50))

pool.close()
pool.join()

df_file = df_file.dropna()
df_file = df_file.drop_duplicates(subset=['sequence1'],keep='first')
df_file = df_file.reset_index(drop=True)
df_file.to_csv(path+'merged_465_entity_uri_entity_sep_google_pre.csv',encoding='utf-8',index=False)
print (df_file.info())
#display(df_file.head(50))
del df_file