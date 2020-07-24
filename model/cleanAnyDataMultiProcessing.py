#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import nltk
from IPython.display import Markdown, display
from unicodedata import normalize
import string
import re
pd.set_option('display.max_colwidth', -1)
import tensorflow as tf
nltk.download('punkt')
import numpy as np
import os
#import urllib2
#from bs4 import BeautifulSoup
#from requests import get
import multiprocessing as mp
import time


# In[ ]:


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
    
    #for dataFrame
    #chunk.loc[:,'wikiAltLabel':'WikidataEntity'] = chunk.loc[:,'Entity':'WikidataEntity'].applymap(applyFunSeries)
    
    #For series
    chunk.loc[:,'wikiAltLabel'] = chunk.loc[:,'wikiAltLabel'].apply(applyFunSeries)
    return chunk

n_pool = mp.cpu_count()
pool = mp.Pool(28)

df_file = pd.DataFrame()
chunks = pd.read_csv('../../dataset/entityData/WikidataAltLabel_clean.csv', dtype= str, encoding='utf-8',usecols=['qValue', 'wikiAltLabel'], chunksize=10000)
print (n_pool,n_pool)

df_file = pd.DataFrame()

for chunk in pool.imap(chunkTaskWorker,chunks, chunksize=1):
    df_file = df_file.append(chunk,ignore_index=True)
    #display(df_file.head(10))

display(df_file.head(50))

pool.close()
pool.join()

df_file = df_file.dropna()
#df_file = df_file.drop_duplicates(subset=['sequence1'],keep='first')
df_file = df_file.reset_index(drop=True)
df_file.to_csv('../../dataset/entityData/WikidataAltLabel_clean_noise.csv',encoding='utf-8',index=False)
print (df_file.info())
display(df_file.head(50))
#del df_file




