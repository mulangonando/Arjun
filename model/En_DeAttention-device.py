#!/usr/bin/env python
# coding: utf-8
#shuffle data after complete epoch instead for taking random pairs for data

# Acknowlegements
# Chatbot Tutorial
================
# **Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`
# 1) Yuan-Kuei Wu’s pytorch-chatbot implementation:
#    https://github.com/ywk991112/pytorch-chatbot
#
# 2) Sean Robertson’s practical-pytorch seq2seq-translation example:
#    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
#
# 3) FloydHub’s Cornell Movie Corpus preprocessing code:
#    https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus

# In[1]:


from __future__ import absolute_import,unicode_literals, print_function, division

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import bcolz
import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


# In[2]:


USE_CUDA = torch.cuda.is_available()
print (USE_CUDA)


device = torch.device("cuda:1" if USE_CUDA else "cpu")
#device = torch.device("cpu")
devices = {}
if torch.cuda.is_available():
    torch.cuda.set_device(1) 
    for i in range(torch.cuda.device_count()):
        devices[i] = torch.device('cuda:'+str(i))
else:
    devices['cpu']= 0

print ("Devices: {}".format(devices))
print ("Current Device: {}".format(torch.cuda.current_device()))

#torch.nn.DataParallel(model, device_ids=[0, 1, 2]) #for parrellel batch processing


# In[3]:


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[4]:


#filter sentence with length and prepareVocabList
'''data = pd.read_csv('../../dataset/entityData/merged_465_entity_uri_entity_google_pre.csv',encoding='utf-8', usecols=['sequence1', 'sequence2','uri', 'uriSequence2'])

print (data.info())

def calculateSentLength(sent):
    sent_tokens = str(sent.encode('utf-8')).split()
    return len(sent_tokens)

data = data.dropna(subset=['sequence1','sequence2', 'uriSequence2'])
print (data.info())
data['seq1len'] = data['sequence1'].apply(lambda sent:calculateSentLength(sent))
data['seq2len'] = data['uriSequence2'].apply(lambda sent:calculateSentLength(sent))
data = data.drop_duplicates(subset=['sequence1'],keep='first')
data = data.reset_index(drop=True)

data.to_csv('../../dataset/entityData/merged_465_entity_uri_entity_google_pre_seq.csv',encoding='utf-8',index=False)

data_Analysis = data[['seq1len','seq2len']]
print (data_Analysis.describe())
data_ranged = data[(data['seq1len']<=25)] #to change
print (data_ranged.describe())
brange = np.arange(1,100,10)
print (brange[0:10])
data_ranged[['seq1len','seq2len']].plot.hist(bins=brange, histtype='bar', alpha=0.5)
data_ranged = data_ranged.reset_index(drop=True)
print (data_ranged.head(5))
data_ranged.to_csv('../../dataset/entityData/merged_465_entity_uri_entity_google_pre_seq_0_25.csv',encoding='utf-8',index=False)
#to change
print (data_ranged.info)
print (data_ranged[['seq1len','seq2len']].describe())


#vocabList building
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = '../../dataset/glove.6B/glove.6B.300d.txt'
word2vec_output_file = '../../dataset/glove.6B/glove.6B.300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
filename = '../../dataset/glove.6B/glove.6B.300d.txt.word2vec'
model_glove = KeyedVectors.load_word2vec_format(filename, binary=False)
vocabList = list(model_glove.vocab.keys())
print (type(vocabList))
np.savetxt('../../dataset/entityData/vocab_glove_full.txt',vocabList,fmt='%s',encoding='utf-8') #to change
del model_glove
vocabDict = {word:value for value, word in enumerate(vocabList)}
del vocabList
seq1Set = set()
seq2Set = set()
chunks_seq1 = pd.read_csv('../../dataset/entityData/merged_465_entity_uri_entity_google_pre_seq_0_25.csv', encoding='utf-8', usecols=['sequence1'], chunksize=10000)
chunks_seq2 = pd.read_csv('../../dataset/entityData/merged_465_entity_uri_entity_google_pre_seq_0_25.csv', encoding='utf-8', usecols=['uriSequence2'], chunksize=10000)

def findVocabSeq1(chunk):
    seqs = chunk['sequence1'].tolist()
    vocabset=[]
    try:
        vocabset = set([word for seq in seqs for word in set(seq.split()) if word in vocabDict])
    except:
        pass
    return vocabset

def findVocabSeq2(chunk):
    seqs = chunk['uriSequence2'].tolist()
    vocabset=[]
    try:
        vocabset = set([word for seq in seqs for word in set(seq.split()) if word in vocabDict])
    except:
        pass
    return vocabset
n_pool = mp.cpu_count()
pools = mp.Pool(28)

for vocabset in pools.map(findVocabSeq1,chunks_seq1):
    seq1Set = seq1Set.union(vocabset)

for vocabset in pools.map(findVocabSeq2,chunks_seq2):
    seq2Set = seq2Set.union(vocabset)

pools.close()
pools.join()
seq3Set = seq1Set.union(seq2Set)
np.savetxt('../../dataset/entityData/vocab_entity_uriEntity_glove_0_25.seq2seq1',list(seq3Set), fmt='%s', encoding='utf-8')
print (len(seq3Set))'''


# In[5]:


batch_size = 128
embedding_size = 300
dropout = 0.2
infer_batch_size = 16
learning_rate =  0.001 #adam
#learning_rate = 0.01 #SGD
clip = 50.0 #adam
#clip = 5.0 #sgd
teacher_forcing_ratio = 1.0
decoder_learning_ratio = 5.0
max_gradient_norm = 5.0
model_name = 'seq2seqEntity_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 256
encoder_n_layers = 1
decoder_n_layers = 1
src_max_len = 50
srx_max_len_infer = 50
tgt_max_len = 50
tgt_max_len_infer = 50
src_voc_words = 0
tar_voc_words = 0
# Default word tokens
EOS = '</s>'
SOS = "<s>"
UNK = '<UNKNOWN>'
NUMBER = '<NUMBER>'
YEAR = '<YEAR>'
SEP = 'wikidataentityseparation'
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3
NUMBER_token = 4
YEAR_token = 5
SEP_token = 6
MAX_LENGTH = 50
corpus_name = 'entityData_Sep'
delimiter = '\t'
pd.set_option('display.max_colwidth', -1)
preTrainembVocabSize1 = 400000
preTrainemb_path1 = '../../dataset/glove.6B/'
preTrainemb_file1= 'glove.6B.300d.txt'
corpus = '../../dataset/trainingData/Dec11_sep/'
vocabList1 = np.loadtxt('../../dataset/entityData/vocab_entity_uriEntity_glove_0_25.seq2seq1', 
                        encoding='utf-8', dtype=str).tolist()
csvFile = '../../dataset/entityData_Sep/merged_465_entity_uri_entity_sep_google_pre_seq_0_25.csv'
datafile = os.path.join(corpus, "formatted_seq_0_25.txt")
DEBUG=False
DEBUG_MODEL=False


# In[6]:


#Create Embedding and Save to files
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "<UNKNOWN>",
                           NUMBER_token: "<NUMBER>",YEAR_token: "<YEAR>", SEP_token: 'wikidataentityseparation'}
        self.num_words = 7  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

class vocab():
    def __init__(self, preTrainemb_path,preTrainemb_file,preTrainembVocabSize,embedding_size, 
                       createEmbeddingMatrixFlag=False,name='glove_300',vocabList=None, debug=False):
        
        self.word2index = {}
        self.word2Vec = {}
        self.words_found = 0
        self.words_not_found = 0
        self.index2word = {}
        self.specialToken = ["PAD","SOS","EOS","<UNKNOWN>","<NUMBER>","<YEAR>", "wikidataentityseparation"]
        self.name = name
        self.embedding_size = embedding_size
        # check this order is correct with token_id (0,1,2,3,4,5)
        self.words= list.copy(self.specialToken)
        self.index = len(self.specialToken)
        self.vocabList = vocabList
        self.preTrainemb_path = preTrainemb_path
        self.preTrainemb_file = preTrainemb_file
        self.preTrainembVocabSize = preTrainembVocabSize
        self.num_words = 0
        self.debug=debug

        
        if not self.vocabList:
            self.matrix_len = self.preTrainembVocabSize+self.index
            self.num_words = len(self.preTrainembVocabSize)+self.index
        else:
            self.matrix_len = len(self.vocabList)+self.index
            self.num_words = len(self.vocabList)+self.index
        
        self.embedding_matrix = np.zeros((self.matrix_len, self.embedding_size))
        
        
        if createEmbeddingMatrixFlag==True:
            self.vectors = bcolz.carray(np.random.normal(scale=0.6, size=(self.index,self.embedding_size)),
                                    rootdir=self.preTrainemb_path+self.name+'_wiki.dat', mode='w')
            self.createEmbeddingMatrix()
        else:
            self.readFiles()
            
    def createEmbeddingMatrix(self):
        self.parseFile()
        if not self.vocabList:
            for i, word in enumerate(self.words):
                try: 
                    self.embedding_matrix[i] = self.word2Vec[word]
                    self.words_found += 1
                except KeyError:
                    self.words_not_found +=1
                    self.embedding_matrix[i] = np.random.normal(scale=0.6, size=(self.embedding_size, ))
                self.word2index[word]=i
                self.index2word[i]=word
        else:
            for i, word in enumerate(self.specialToken + self.vocabList):
                try:
                    if self.debug==True: print (i, word, self.word2Vec[word].shape)
                    self.embedding_matrix[i] = self.word2Vec[word]
                    self.words_found += 1
                except KeyError:
                    self.words_not_found +=1
                    self.embedding_matrix[i] = np.random.normal(scale=0.6, size=(self.embedding_size, ))
                self.word2index[word]=i
                self.index2word[i]=word
            
        self.saveFiles()
        self.readFiles()
             
        
    def parseFile(self):
        with open(self.preTrainemb_path+self.preTrainemb_file, 'rb') as f:
            for l in f:
                try:
                    line = l.decode().split()
                    word = line[0]
                    self.words.append(word)
                    vect = np.array(line[1:]).astype(np.float)
                    self.vectors.append(vect)
                except:
                     print ("########### Error while parsing preTrainemb file ###########")
        self.vectors.flush()
        self.getword2Vec()
                
    
    def saveFiles(self):
        #vectors = bcolz.carray(self.vectors[1:].reshape((preTrainembVocabSize, self.embedding_size)), rootdir=preTrainemb_path+preTrainembFilePre+'.dat', mode='w')
        #vectors.flush()
        pickle.dump(self.embedding_matrix,open(self.preTrainemb_path+self.name+'_embMat_wiki.pkl', 'wb') )
        pickle.dump(self.words, open(self.preTrainemb_path+self.name+'_words_wiki.pkl', 'wb'))
        pickle.dump(self.word2index, open(self.preTrainemb_path+self.name+'_w2i_wiki.pkl', 'wb'))
        pickle.dump(self.index2word, open(self.preTrainemb_path+self.name+'_i2w_wiki.pkl', 'wb'))
    
    def readFiles(self):
        self.vectors = bcolz.open(self.preTrainemb_path+self.name+'_wiki.dat')[:]
        self.words = pickle.load(open(self.preTrainemb_path+self.name+'_words_wiki.pkl', 'rb'))
        self.word2index = pickle.load(open(self.preTrainemb_path+self.name+'_w2i_wiki.pkl', 'rb'))
        self.index2word = pickle.load(open(self.preTrainemb_path+self.name+'_i2w_wiki.pkl', 'rb'))
        self.embedding_matrix = pickle.load(open(self.preTrainemb_path+self.name+'_embMat_wiki.pkl', 'rb') )
    
    def getword2Vec(self):
        self.word2Vec = {w: self.vectors[i] for i,w in enumerate(self.words)}
        return self.word2Vec
    
    def getStats(self):
        print ("word2index, index2word, words")
        return self.word2index, self.index2word
    
    def getEmbeddingMatrix(self):
        return self.embedding_matrix
    
    def  printVocabStat(self):
         print ("word2index: {} index2word: {}".format(len(self.word2index), len(self.index2word)))


#voc = vocab(preTrainemb_path1,preTrainemb_file1,preTrainembVocabSize1,embedding_size, True, 'glove_300',vocabList1)
#voc = vocab(preTrainemb_path1,preTrainemb_file1,preTrainembVocabSize1,embedding_size, False, 'glove_300',vocabList1)


# In[7]:


#reading data from CSV file NTBC if changing reading file
#data = pd.read_csv(csvFile,encoding='utf-8')
#print (data.info())
# need to change if you want new file here NTBC
#data.to_csv(datafile, header=None, index=None, sep=delimiter, encoding='utf-8')


# In[8]:


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').        read().strip().split('\n')
    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[s for s in l.split('\t')] for l in lines]
    #voc = Voc(corpus_name) # need to change if not using glove or any pretrained Vocab NTBC
    #to change
    voc = vocab(preTrainemb_path1,preTrainemb_file1,preTrainembVocabSize1,embedding_size, True, 'glove_300',vocabList1)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[1].split(' ')) < MAX_LENGTH and len(p[3].split(' ')) < MAX_LENGTH and len(p[5].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    #Need to change for if not using glove or any pretrained Vocab -NTBC
    '''for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])'''
    print("Counted words:", len(voc.words))
    return voc, pairs


# Load/Assemble voc and pairs
#sampledatafile = '../../dataset/trainingData/oct_10/formatted_seq_0_50_sample.txt'
save_dir = os.path.join(corpus, "save")
#voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
#print some pairs to validate
print("\npairs:")
#for pair in pairs[:10]:
#    print(pair.enocde('utf-8'))
print ("Number of Pairs: {}".format(len(pairs)))


# In[9]:


#repalcing unknown word with token
def replaceWord2index(voc, word):
    try:
        index = voc.word2index[word]
    except:
        index = UNK_token
    return index

def indexesFromSentence(voc, sentence):
    return [replaceWord2index(voc,word) for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[1].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        #print (pair[1].encode('utf-8'), pair[5].encode('utf-8'))
        input_batch.append(pair[1])
        output_batch.append(pair[5]) #need to be change according to dataFile
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
#here we can pass directly pairs from csv file and voc should be ready from Vocab
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


# In[10]:


#Embedding Layer for Pretrained Embeddings
def create_Embedding(embedding_matrix, non_trainable=False):
    num_embeddings, embedding_dim = embedding_matrix.shape[0], embedding_matrix.shape[1]
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    embedding.load_state_dict({'weight': embedding_matrix})
    if non_trainable:
        embedding.weight.requires_grad = False

    return embedding, num_embeddings, embedding_dim


# In[11]:


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, embedding, n_layers=1, dropout=0,debug=False):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        #self.gru = nn.LSTM(hidden_size, hidden_size, n_layers,
        #                  dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout=(0 if n_layers == 1 else dropout), batch_first = False, bidirectional=True)
        self.debug=debug
    
    # input: seq_len, batch, input_size
    # 
    def forward(self, input_seq, input_lengths, device, hidden=None):
        input_seq, input_lengths = input_seq.to(device), input_lengths.to(device)
        if hidden == None:
            pass
        else:
            hidden = hidden.to(device)
        if self.debug==True: print ("EncoderRnn0: {}{}{}".format(input_seq.device, input_lengths.device, hidden.device))
        # Convert word indexes to embeddings
        if self.debug==True: print ("EncoderRnn1: {}".format(input_seq.size()))
        embedded = self.embedding(input_seq)
        if self.debug==True: print ("EncoderRnn2: {}".format(embedded.size()))
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        if self.debug==True: print ("EncoderRnn3: {}".format(packed.data.size()))
        # Forward pass through LSTM
        outputs, hidden = self.lstm(packed, hidden)
        if self.debug==True: print ("EncoderRnn4: {} {} {}".format(outputs.data.size(),hidden[0].size(),hidden[1].size()))
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.debug==True: print ("EncoderRnn5: {}".format(outputs.data.size()))
        # Sum bidirectional LSTM outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        if self.debug==True: print ("EncoderRnn6: {}".format(outputs.data.size()))
        # Return output and final hidden state
        # outputs: (sequencesize, batchsize, hiddensize), hidden: (bidirectional, batchsize, hiddensize)
        return outputs, hidden

'''EncoderRnn1: torch.Size([9, 4])
EncoderRnn2: torch.Size([9, 4, 300])
EncoderRnn3: torch.Size([28, 300])
EncoderRnn4: torch.Size([28, 512]) torch.Size([2, 4, 256]) torch.Size([2, 4, 256])
EncoderRnn5: torch.Size([9, 4, 512])
EncoderRnn6: torch.Size([9, 4, 256])'''


# In[12]:


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size,debug=False):
        super(Attn, self).__init__()
        self.method = method
        self.debug=debug
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
    
    
    def dot_score(self, hidden, encoder_output):
        hidden, encoder_output = hidden.to(devices[1]), encoder_output.to(devices[1])
        if self.debug==True: print ("Attn2: {}".format(torch.sum(hidden * encoder_output, dim=2).size()))
        #return all states scores withrespect to target state for all batches
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        hidden, encoder_output = hidden.to(devices[1]), encoder_output.to(devices[1])
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        hidden, encoder_output = hidden.to(devices[1]), encoder_output.to(devices[1])
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        hidden, encoder_outputs = hidden.to(devices[1]), encoder_outputs.to(devices[1])
        # Calculate the attention weights (energies) based on the given method
        if self.debug==True: print ("Attn1: {} {}".format(hidden.size(),encoder_outputs.size()))
        
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        if self.debug==True: print ("Attn3: {}".format(attn_energies.size()))
        attn_energies = attn_energies.t()
        if self.debug==True: print ("Attn4: {}".format(attn_energies.size()))

        # Return the softmax normalized probability scores (with added dimension)
        if self.debug==True: print ("Attn5: {}".format(F.softmax(attn_energies, dim=1).unsqueeze(1).size()))
        #it is softmax of scores for all states given all batches
        #basically we have a_t of all states
        #size (batchsize, 1, seqSize)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# In[13]:


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, embedding_size, hidden_size, output_size, n_layers=1, dropout=0.1,debug=False):
        #here output_size is decoder vocabsize
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.debug=debug

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs, device):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        input_step, encoder_outputs = input_step.to(device),encoder_outputs.to(device)
        last_hidden = (last_hidden[0].to(device), last_hidden[1].to(device))
        if self.debug==True: print ("LuongAttnDecoderRNN0: {} {} {}".format(input_step.device,last_hidden[0].device, 
                                                          encoder_outputs.device))
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        if self.debug==True: print ("LuongAttnDecoderRNN1: {} {}".format(embedded.size(),last_hidden[1].size()))
        # Forward through unidirectional LSTM
        rnn_output, hidden = self.lstm(embedded, last_hidden)
        if self.debug==True: print ("LuongAttnDecoderRNN2: {} {}".format(rnn_output.size(), hidden[0].size()))
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        attn_weights = attn_weights.to(device)
        if self.debug==True: print ("LuongAttnDecoderRNN3: {}".format(attn_weights.size()))
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        if self.debug==True: print ("LuongAttnDecoderRNN4: {}".format(encoder_outputs.size()))
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        if self.debug==True: print ("LuongAttnDecoderRNN5: {}".format(context.size()))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        if self.debug==True: print ("LuongAttnDecoderRNN6: {}{}".format(rnn_output.size(), context.size()))
        concat_input = torch.cat((rnn_output, context), 1)
        if self.debug==True: print ("LuongAttnDecoderRNN7: {}".format(concat_input.size()))
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        if self.debug==True: print ("LuongAttnDecoderRNN8: {}".format(concat_output.size()))
        output = self.out(concat_output)
        if self.debug==True: print ("LuongAttnDecoderRNN9: {}".format(output.size()))
        output = F.softmax(output, dim=1)
        if self.debug==True: print ("LuongAttnDecoderRNN10: {}".format(output.size()))
        # Return output and final hidden state
        return output, hidden


# In[14]:


#This loss function itsself need to be changed as it only takes log of softmax but does not compute any loss like
# meanSquare or crossEntropy loss 
# it can be justified as log(1.0)==0 and log(0.0)=inf
#target there will be a single batch without any index just padded value 0 so we need to consider mask here
#in target (having word index) you want to take data, gather collects data from input accoording to target indexes for loss
def maskNLLLoss(inp, target, mask,debug=False):
    inp = inp.to(devices[1])
    target = target.to(devices[1])
    mask = mask.to(devices[1])
    if debug==True: print ("maskNLLLoss:{}{}{}".format(inp.size(),target,mask))
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    #mask should be mask.view(-1,1) : NTBM
    loss = crossEntropy.masked_select(mask)
    if debug==True: print ("maskNLLLoss:{}".format(loss.size()))
    # if we change mean here is due to mask_size, it should be releated to batch_size()
    lossMean = loss.mean()
    lossMean = lossMean.to(devices[1])
    return lossMean, nTotal.item()


# In[15]:


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding1,embedding2,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH, debug=False):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    if debug==True: print ("train1:{}{}{}{}".format(input_variable.size(),lengths.size(),target_variable.size(), mask.size()))
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, devices[1])
    if debug==True: print ("train2:{}{}".format(encoder_outputs.size(),encoder_hidden[0].size()))
    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    if debug==True: print ("train3:{}".format(decoder_input.size()))
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = (encoder_hidden[0][:decoder.n_layers],encoder_hidden[1][:decoder.n_layers])
    if debug==True: print ("train3.1:{}".format(decoder_hidden[0].size()))
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, devices[2]
            )
            if debug==True: print ("train4:{}{}{}".format(decoder_output.size(),decoder_input.size(),decoder_hidden[0].size()))
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            if debug==True: print ("train5:{}".format(decoder_input.size()))
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            if debug==True: print ("train6:{}{}".format(loss,n_totals))
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, devices[2]
            )
            if debug==True: print ("train7:{}{}".format(decoder_input.size(),decoder_hidden.size()))
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            if debug==True: print ("train8:{}".format(decoder_input.size()))
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            if debug==True: print ("train9:{}{}".format(loss,n_totals))

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


# In[16]:


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding1, embedding2,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, save_every_epoch, plot_every, plot_losses, clip, corpus_name, loadFilename, epoch,start_iteration=1,debug=False):
    
    start = time.time()
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]
    
    if debug==True: print ("trainIters1: {}".format(training_batches[0]))
    # Initializations
    print('Initializing ...')
    print_loss = 0
    plot_loss = 0

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        if debug==True: print ("trainIters2: {}".format(training_batch))
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        #print (input_variable, lengths, target_variable, mask, max_target_len)
        
        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding1, embedding2, encoder_optimizer, decoder_optimizer, batch_size, clip, debug=False)
        print_loss += loss
        plot_loss  += loss

        # print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Time {} Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(timeSince(start, iteration / n_iteration), 
                                                                                          iteration, iteration / n_iteration * 100,print_loss_avg))
            print_loss = 0
        
        #plot progress
        if iteration % plot_every == 0:
            plot_loss_avg = plot_loss / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss = 0

        # Save checkpoint
        if ((epoch % save_every_epoch) and  (iteration % save_every) == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'epoch':epoch,
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding1': embedding1.state_dict(),
                'embedding2': embedding2.state_dict()
            }, os.path.join(directory, '{}_{}_{}.tar'.format(epoch, iteration, 'checkpoint')))


# In[17]:


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, debug=False):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.debug=debug

    def forward(self, input_seq, input_length, max_length, device):
        # Forward input through encoder model
        input_seq, input_length = input_seq.to(device), input_length.to(device)
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length, device)
        if self.debug==True: print ("GreedySearchDecoder1: {} {}".format(encoder_outputs.size(),encoder_hidden.size()))
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        #is it we can do something here ??? NTBC
        decoder_hidden = (encoder_hidden[0][:decoder.n_layers],encoder_hidden[1][:decoder.n_layers])
        if self.debug==True: print ("GreedySearchDecoder2: {}".format(decoder_hidden[0].size()))
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        decoder_input = decoder_input.to(device)
        if self.debug==True: print ("GreedySearchDecoder3: {}".format(decoder_input.size()))
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, device)
            if self.debug==True: print ("GreedySearchDecoder4: {} {}".format(decoder_output.size(),decoder_hidden[0].size()))
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            if self.debug==True: print ("GreedySearchDecoder5: {} {}".format(decoder_scores,decoder_input.size()))
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            if self.debug==True: print ("GreedySearchDecoder6: {} {}".format(all_tokens,all_scores))
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            if self.debug==True: print ("GreedySearchDecoder7: {}".format(decoder_input.size()))
        # Return collections of word tokens and scores
        return all_tokens, all_scores


# In[18]:


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH, debug=False):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    if debug==True: print ("evaluate: {} {}".format(indexes_batch,lengths))
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length, devices[1])
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def word_accuracy(tar_sent, pred_sent):
    acc = 0.
    labels = tar_sent.strip().split(" ")
    preds = pred_sent.strip().split(" ")
    match = 0.0
    match = sum([1 for pos in range(min(len(labels), len(preds))) if labels[pos]==preds[pos]])
    acc += 100 * match / max(len(labels), len(preds))
    return acc

def evaluateInput(encoder, decoder, searcher, voc, pair):
        output_words = evaluate(encoder, decoder, searcher, voc, pair[1])
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        #print('PredictedEntity:', ' '.join(output_words))
        #print (pairs[1])
        #print (' '.join(output_words))
        tar_sent = pair[5]
        pred_sent = ' '.join(output_words)
        #sur_sent = pair[1]
        #q_values = pair[3]
        #df = pd.DataFrame()
        #d = {'OSeq':pair[0].strip(),'TSeq':tar_sent, 'PSeq':pred_sent, 'uri':q_values, 'SSeq':sur_sent}
        #df = pd.DataFrame(data=d, index=[0])
        #global df_total
        #df_total = df_total.append(df,ignore_index=True)
        return word_accuracy(tar_sent, pred_sent)
        


# In[19]:


# Configure models

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
starting_epoch = 1
start_iteration = 1
plot_losses = [1.0]


#checkpoint_iter = '7_7096'
#plot_losses = np.load('epoch_loss_Nov13_sur.npy')
#change epoch Number too
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    starting_epoch = checkpoint['epoch']
    start_iteration = checkpoint['iteration'] + 1
    print ("Loading checkpoint fileName: {}".format(loadFilename))
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd1 = checkpoint['embedding1']
    embedding_sd2 = checkpoint['embedding2']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
#NTBC according to voc
print (voc.embedding_matrix.shape[0], voc.embedding_matrix.shape[1])
#non-trainable or trainable vocablury
embedding1, num_embeddings2, embedding_dim2 = create_Embedding(torch.from_numpy(voc.embedding_matrix), False)
embedding2, num_embeddings2, embedding_dim2 = create_Embedding(torch.from_numpy(voc.embedding_matrix), False)
#embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding1.load_state_dict(embedding_sd1)
    embedding2.load_state_dict(embedding_sd2)
# Initialize encoder & decoder models
#embedding = embedding
encoder = EncoderRNN(hidden_size, embedding_size, embedding1, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding2, embedding_size, hidden_size, voc.num_words, decoder_n_layers, dropout, False)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(devices[1])
decoder = decoder.to(devices[2])
print('Models built and ready to go!')


# In[20]:


#%time
# Configure training/optimization
#2136960
n_iteration = int(len(pairs)/batch_size)
print_every = int(n_iteration/40)
plot_every = int(n_iteration/200)
save_every = int(n_iteration)  #every_epoch
save_every_epoch = int(2)
epochs = 15

#sample test Value
'''n_iteration = 4
print_every = 1
save_every = 2
plot_every = 1
save_every_epoch = int(2)
epochs = 5
batch_size=128'''


# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
#decay learning rate
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

#encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)
#decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, momentum=0.9)


encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=2, factor=0.5, verbose=True)
decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=2, factor=0.5, verbose=True)

if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
#devide pairs in Train, dev and test
#random.shuffle(pairs) not needed because for comparision

# train, dev, test ratios
total = len(pairs)
train_size = int(80*total/100)
dev_size= int(19*total/100)
test_size = int(1*total/100)
print (total, train_size, dev_size, test_size)
trainPairs, devPairs, testPairs =pairs[0:train_size],pairs[train_size:train_size+dev_size],pairs[train_size+dev_size:-1] 

#sample Test
#trainPairs, devPairs, testPairs = trainPairs[0:200], devPairs[0:10], testPairs[0:10]
print("Starting Training!")
# Run training iterations
print ('starting_epoch: {}, start_iteration: {}'.format(starting_epoch, start_iteration))
for epoch in range(starting_epoch,epochs):
    print ('Starting Epoch Number: {} for training'.format(epoch))
    decoder = decoder.to(devices[2])
    trainIters(model_name, voc, trainPairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
          embedding1, embedding2, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, save_every_epoch, plot_every, plot_losses, clip, corpus_name, loadFilename, epoch, start_iteration)
    start_iteration = 1
    # Initialize search module
    encoder_scheduler.step(plot_losses[-1])
    decoder_scheduler.step(plot_losses[-1])
    #Need to be change :NTBC
    if (epoch%4==0):
        print("Evaluation Start!")
        encoder.eval()
        decoder.eval()
        start = time.time()
        decoder = decoder.to(devices[1])
        searcher = GreedySearchDecoder(encoder, decoder).to(devices[1])
        
        
        #need to be changing For Train Accuracy
        #accuracies_train = [evaluateInput(encoder, decoder, searcher,voc, pair) for pair in trainPairs[0:2]]
        #accuracy_train = sum(accuracies_train)/len(accuracies_train)
        #print("Time: {} Epoch: {}; Average train accuracy: {:.4f}".format(timeSince(start, epoch / epochs),epoch,accuracy_train))
            
        accuracies_dev =  [evaluateInput(encoder, decoder, searcher, voc, pair) for pair in devPairs]
        accuracies_test = [evaluateInput(encoder, decoder, searcher,voc, pair) for pair in testPairs]
    # Set dropout layers to eval mode
        accuracy_dev = sum(accuracies_dev)/len(accuracies_dev)
        accuracy_test = sum(accuracies_test)/len(accuracies_test)
        print("Time: {} Epoch: {}; Average dev accuracy: {:.4f}; Average test accuracy: {:.4f}".format(timeSince(start, epoch / epochs),epoch,accuracy_dev, accuracy_test))
        
        #global df_total
        #df_total.to_csv(save_dir+epoch+'predicted.csv',encoding='utf-8',index=False)
        
        encoder.train()
        decoder.train()
    with open('epoch_loss_Dec11_sep.npy','ab') as f:   #NTBC
        np.save(f, np.array(plot_losses))
    #print (plot_losses)
showPlot(plot_losses)
f.close()
print ("------------------Finished training---------------------")
