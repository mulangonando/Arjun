
# coding: utf-8

# In[1]:


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
import urllib2
from bs4 import BeautifulSoup
from requests import get


# In[2]:


display("Hello To CompleteParser")
df_file = pd.DataFrame()
path = '../../dataset/Trex/'
doc_id = 0


# In[3]:


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
    
    
def parseDoc(sentences, entityJson, sentences_boundaries):
    global df_file
    global doc_id
    #sentence_list = sent_tokenize(sentences)
    entity_list_dict = {entity['boundaries'][0]:entity['surfaceform'] for entity in entityJson 
                       if  entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker' 
                        #or entity['annotator'] == 'Wikentity_list_dict = {entity['boundaries'][0]:entity['surfaceform'] for entity in entityJson 
                       if  entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker' 
                        #or entity['annotator'] == 'Wikidata_Property_Linker'
                        }
    entity_list_dict_uri = {entity['boundaries'][0]:entity['uri'] for entity in entityJson 
                       if  entity['annotator'] == 'Wikidata_Spotlight_Entity_Linker' 
                        #or entity['annotator'] == 'Wikidata_Property_Linker'
                        }
    entity_list = [entity_list_dict[key] for key in sorted(entity_list_dict.iterkeys())]
    entities = ' '.join(entity for entity in entity_list)
    
    entity_list_uri = [entity_list_dict_uri[key].replace('http://www.wikidata.org/entity/', '') for key in sorted(entity_list_dict_uri.iterkeys())]
    entities_uri = ' '.join(entity for entity in entity_list_uri)

    #code for if we want to change text to list of sentences
    '''for b in sentences_boundaries:
        #print (sentences[b[0]:b[1]])
    #for sentence in sentence_list:
        sentence = sentences[b[0]:b[1]]
        sentence2 = searchEntityinSentence(entity_list, sentence)
        d = {'sequence1':sentence, 'sequence2':sentence2}
        df = pd.DataFrame(data=d, index=[0])
        df_file = df_file.append(df,ignore_index=True)'''
    if len(entity_list_uri)==len(entity_list_uri):
        #entity_uriEntity = {entity:extractEntityFromUri(entityUri) for entity, entityUri in zip(entity_list, entity_list_uri)}
        #print entity_uriEntity
        #d = {'docid':doc_id,'sequence1':sentences, 'sequence2':entities, 'uri':entities_uri, 'entity:uriEntity':entity_uriEntity}
        d = {'docid':doc_id,'sequence1':sentences, 'sequence2':entities, 'uri':entities_uri}
        df = pd.DataFrame(data=d, index=[0])
        #df = pd.DataFrame(data=d)
        df = df.fillna('')
        df_file = df_file.append(df,ignore_index=True)

        del df
    else:
        print (doc_id)
    doc_id = doc_id + 1
    

def paserFile(n_file, filename):
    df = pd.read_json(filename,encoding='utf-8')
    df = df[[u'text',u'entities', u'sentences_boundaries']]
    for i in range(len(df.index)):
        parseDoc(df.iloc[i,0], df.iloc[i,1], df.iloc[i,2])
    del df


# In[4]:


#filesList = sorted(os.listdir(path+'Trex/')) #to change
filesList = sorted(os.listdir(path))
#print (sorted(filesList))
for n_file, file_name in enumerate(filesList): #to change
    paserFile(n_file, path+file_name)
    print (df_file.info())

path = '../../dataset/'

df_file = df_file.dropna()
df_file = df_file.drop_duplicates(keep='first')
df_file = df_file.reset_index(drop=True)
df_file.to_csv(path+'merged_465_entity'+'.csv',encoding='utf-8',index=False)
print (df_file.info())
display(df_file.head(50))
del df_file


# In[5]:


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

    


# In[28]:


'''df_file = pd.DataFrame()
chunks = pd.read_csv(path+'merged_465_entity'+'.csv', encoding='utf-8',usecols=['docid','sequence1', 'sequence2'], chunksize=10000)
for i, chunk in enumerate(chunks):
    #try:
    chunk = chunk.dropna()
    chunk.loc[:,'sequence1'].map(lambda x:textToWordTokenTotext(x))
    #display(chunk)
    chunk['sequence1'] = chunk['sequence1'].map(lambda x:textToWordSeqGoogle(x))
    chunk['sequence1'] = chunk['sequence1'].map(lambda x:textTotoken(x))
    #display(chunk)
    #chunk = chunk.applymap(removeGoogleStopWords)
    chunk['sequence1'] = chunk['sequence1'].map(lambda x:replaceYears(x))
    chunk['sequence1'] = chunk['sequence1'].map(lambda x:replaceNumbers(x))
    #display(chunk)
    chunk['sequence1'] = chunk['sequence1'].map(lambda x:removeAlphaNum(x))
    #display(chunk)
    chunk['sequence1'] = chunk['sequence1'].map(lambda x:tokenTotext(x))
    display(chunk.head(10))
    #df_file = df_file.append(chunk,ignore_index=True)
    #except:
    #    print i,
    #    display(chunk)
        #chunk = chunk.applymap(nonVectorizedGoogleWords)
    #display(chunk)
df_file = df_file.dropna()
df_file = df_file.drop_duplicates(subset=['sequence1', 'sequence2'],keep='first')
df_file = df_file.reset_index(drop=True)    
df_file.to_csv(path+'merged_465_entity_google_prep.csv',encoding='utf-8',index=False)
print (df_file.info())
display(df_file.head(50))
del df_file'''

