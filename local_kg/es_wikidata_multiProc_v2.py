#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:25:34 2017

@author: mulang
"""
from pprint import pprint
import editdistance
import os, sys
import csv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
import gc
import multiprocessing as mp
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s --' % (asMinutes(s))

def index_bulk_labels():
    labelsFile = 'WikidataLabel_clean_noise.csv'

    wikidataFolder = os.path.abspath(os.path.join('/home/vyas/dataset/', 'entityData'))

    #We first index the Labels
    with open(wikidataFolder+"/"+labelsFile, 'r') as f: #Add os.pathseperator
        rows = csv.reader(f)
        for row in rows :
            yield {
                "_index": "wikiindexlabels",
                "_type": "document",
                "doc": {"qVal": row[0],
                        'label' : row[1]
                        },
            }
        del rows
        gc.collect()

    f.close()

def index_bulk_knownAs():
    altsFile = 'WikidataAltLabel_clean_noise.csv'

    wikidataFolder = os.path.abspath(os.path.join('/home/vyas/dataset/', 'entityData'))

    # We first index the Labels
    with open(wikidataFolder + "/" + altsFile, 'r') as f:  # Add os.pathseperator
        rows = csv.reader(f)
        for row in rows:
            yield {
                "_index": "wikiindexaltlabels",
                "_type": "document",
                "doc": {"qVal": row[0],
                        'knownAs': row[1]
                        },
            }
        del rows
        gc.collect()

    f.close()

    # print("Total count : ",k)


def entitySearch(line):
    query = line[0].strip()
    es = Elasticsearch("http://eis-warpcore-01:49200")
    #print (es.count())
    indexName = ["wikiindexlabels","wikiindexaltlabels"]
    docType = "document"
    entities = []


    elasticResults = es.search(index=indexName, doc_type=docType, body=
	{"query": {

            "bool": {
                "must": {
                    "bool": {"should": [
                        {"multi_match": {"query": query.strip(),
                                         "fields": ["doc.label"], "fuzziness": "AUTO"}},
                        {"multi_match": {"query": query.strip(), "fields": ["doc.knownAs"], "fuzziness": "AUTO"}},
                    ]}
                }
            }
        }
        , "size": 20
    })

    #print(elasticResults)
    for doc in elasticResults['hits']['hits']:
        # label = doc['_source']['doc']['label']
        keys = list(doc['_source']['doc'].keys())
        #print (keys)
        if keys[1] == 'label' or keys[0] == 'label':
            label = doc['_source']['doc']['label']
        else:
            label = doc['_source']['doc']['knownAs']
            # print("Known As : ",label)

        score = doc['_score']
        qvalue = doc['_source']['doc']['qVal']

        score = lev_similarity(query, label) * score

        i=0

        while i< len(entities):
            if entities[i][2]< score:
                # print([label,qvalue,score])
                entities.insert(i,[label,qvalue,score])

                break

            i = i+1

        if i==len(entities):
            # print([label,qvalue,score])
            entities.append([label,qvalue,score])


    return (entities,line)




def evaluateWiki():
    qs_file = os.path.abspath(os.path.join("/home/vyas/dataset/", "entityData_Sep/surfaceForm_wikiEntity_0_25_noiseclean.csv"))
    all_questions = []
    with open(qs_file, "r", encoding='utf-8') as qf:
        qf_loaded = csv.reader(qf)
        for i,line in enumerate(qf_loaded):
            all_questions.append([line[0].strip(),line[1].strip()])
    qf.close()

    correct_count = 0
    eval_res = []
    
    
    with open("/home/vyas/dataset/entityData_Sep/elasticsearch_proc_ahmed.csv", "wb") as f:
        print ('Creating File')
    
    #for line in all_questions[1:]:
    #    res = entitySearch(line[0].strip())

    #    if len(res)>0:
    #        line.extend([val[1] for val in res[:3]])
    #        eval_res.append(line)

     #   with open("/home/vyas/dataset/entityData_Sep/elasticsearch.csv", "ab+") as f:
     #       wr = ",".join(line)+"\n"
     #       f.write(wr.encode('utf-8'))
     #   f.close()
    
    n_pool = mp.cpu_count()
    pool = mp.Pool(28)

    for res in pool.imap(entitySearch,all_questions[1:], chunksize=1):
        #print (len(res[0]), len(res[1]))
        if len(res[0]) == 1:
            res[1].extend([val[1] for val in res[0][:1]])
            eval_res.append(res[1])
        elif len(res[0]) >= 3:
            res[1].extend([val[1] for val in res[0][:3]])
            eval_res.append(res[1])

        with open('/home/vyas/dataset/entityData_Sep/elasticsearch_proc_ahmed.csv', "ab+") as f:
            wr = ",".join(res[1])+"\n"
            f.write(wr.encode('utf-8'))
        f.close()


    pool.close()
    pool.join()


#This function simply compares the Levenstein distances
def lev_similarity(query, field):

    numerator = 20
    lenth = len(query)

    dist_stems = editdistance.eval(query, field)
    denominator = float((dist_stems+1) * lenth)

    return (numerator/denominator)




if __name__ == '__main__':
    es = Elasticsearch("http://eis-warpcore-01:49200")
    print (es.count())
    
    #bulk(es, index_bulk_labels())
    #bulk(es, index_bulk_knownAs())
    print ("Starting evaluating wiki")
    start = time.time()
    evaluateWiki()
    print("Time: {}".format(timeSince(start)))
    print ("Finished evaluating wiki")