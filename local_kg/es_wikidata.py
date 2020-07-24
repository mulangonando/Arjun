#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:25:34 2017

@author: mulang
"""
from pprint import pprint
# !/usr/bin/env python
# import urllib2
# import urllib
# import json
# import pprint
# import codecs
# import csv
# import sparql
# from SPARQLWrapper import SPARQLWrapper, JSON
# import time
# import pickle as pic
# import re
import editdistance
import os, sys
import csv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
import gc

proj_dir = os.path.abspath(os.path.join('.'))
dbo_file = os.path.abspath(os.path.join('.', 'dbpedia', 'dbo_final'))
sys.path.append(proj_dir)
dbp_file = os.path.abspath(os.path.join('.', 'dbpedia', 'dbp_final'))
dbo_binaries = os.path.abspath(os.path.join('.', 'dbpedia', 'onto_binaries_original'))
dbp_binaries = os.path.abspath(os.path.join('.', 'dbpedia', 'dbp_binaries'))
dbo_text_file = os.path.abspath(os.path.join('.', 'props_lists', 'dbo.csv'))
dbp_text_file = os.path.abspath(os.path.join('.', 'props_lists', 'dbp.csv'))

triples_folder = os.path.abspath(os.path.join('.', 'triples'))
abstarcts_folder = os.path.abspath(os.path.join('.','abstracts'))
# dbp_text_file =

# Elasticsearch configs
dbpedia_props_index = "dbpedia-props"
es_host = "localhost"
es_port = "9200"

# context = create_default_context(cafile="path/to/cert.pem")
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es = Elasticsearch("http://eis-warpcore-01:49200")
print (es.count())

# es = Elasticsearch(
#     ['localhost', 'otherhost'],
#     http_auth=('user', 'secret'),
#     scheme="https",
#     port=9200,
# )

# res = es.index(index="test-index", doc_type='tweet', id=1, body=doc)
# print(res['result'])
#
# res = es.get(index="test-index", doc_type='tweet', id=1)
# print(res['_source'])
#
# es.indices.refresh(index="test-index")
#
# res = es.search(index="test-index", body={"query": {"match_all": {}}})
# print("Got %d Hits:" % res['hits']['total'])
# for hit in res['hits']['hits']:
#     print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])

# def process_abstracts():
#     abs_files = os.listdir(abstarcts_folder)
#     i=0
#     for abs_f in abs_files[0:2]  :
#         abs_triples = []
#         with open(abstarcts_folder+"/"+abs_f, 'r') as f: #Add os.pathseperator
#             rows = csv.reader(f)
#             k=1
#             for row in rows:
#                 if row[0].strip()[0:4].strip() == 'http':
#                     abs_triples.append(row)
#
#                     if abs_triples[-1][-1].find('*')>-1:
#                         print("\n line " + abs_triples[-1][-1])
#                 else:
#                     abs_triples[-1] = abs_triples[-1]+" "+' '.join(part.stip() for part in row)
#
#                     print("\n line "+abs_triples[-1])
#                     print("\n\n")
#
#
#
#             f.close()
#         del abs_triples
#         gc.collect()
#
#         i = i+1
#
#         if i>2:
#             break

# class property


def search_es(nindex,tdoc,sstr):
    # query = {"query": {"match_phrase":{"uri": sstr}}}

    filed=''
    if tdoc == 'triple':
        field = 'subject'
    else:
        field = 'uri'

    query = {"query": {"wildcard": {"subject" : "*Tonight_Show_Starring_Johnny*"}}}
    # props_query = {"query": {"wildcard": {""+filed: "*"+sstr+"*"}}}
    # triples_query = {"query": {"wildcard": {""+filed: "*" + sstr + "*"}}}

    res = es.search(index="dbpedia_triples", doc_type="triple",body=query) #Check subject or Object
    # res = "multi_match": {"query": "guide","fields": ["_all"]}
    # res = es.search(index=nindex, doc_type=tdoc,body=query) #Return the property form ES
    # res = es.search(index=nindex, doc_type=tdoc, body=query)  # Return the property form ES

    print("%d documents found" % res['hits']['total'])
    for doc in res['hits']['hits']:
        #doc keys --> ['_index', '_type', '_id', '_score', '_source']
        # print("The Source thing looks like this",doc['_source'].keys())
        #The Source thing looks like this dict_keys(['uri', 'domain', 'range', 'label', 'comment', 'synonyms', 'phrases', 'annotations', 'num_subjects', 'num_objects'])

        if tdoc == 'triple':
            print("%s %s %s" % (doc['_id'],doc['_source']['subject'],doc['_source']['object']))
        else:
            print("%s %s %s %s %s" % (doc['_id'],doc['_source']['uri'],doc['_source']['domain'],doc['_source']['range'],doc['_source']['synonyms']))


def search_prop(sstr,sstr_full):

    query1 = {"size": 10,"query": {
                    "bool": {"must": {"bool":{"should":[
                        {"multi_match": {"query": sstr_full, "fields": ["label","synonyms"]}},
                        {"wildcard": {"uri": "*" + sstr_full.replace(" ", "")+"*"}}]}}}
        }}

    query2 = {"size": 10,"query": {"wildcard": {"label": "*" + sstr + "*"}}}

    res1 = es.search(index="properties", doc_type="dbo-relation", body=query1,request_timeout=20)
    res2 = es.search(index="properties", doc_type="dbo-relation", body=query2,request_timeout=20)

    dbo_prefix = 'http://dbpedia.org/ontology/'
    dbp_prefix = 'http://dbpedia.org/property'

    props = {}
    uniq_labels = []
    dbo_found=False
    dbp_found=False

    for doc in res1['hits']['hits']:
        uri = doc['_source']['uri']
        if uri not in props.keys() :
            props[uri] = {'label':doc['_source']['label'],'domain':doc['_source']['domain'],'range':doc['_source']['range'],'synonyms':doc['_source']['synonyms']}
            uniq_labels.append(uri.split("/")[-1])

        prop_type=uri.split("/")[-2]

        if prop_type=="ontology":
            dbo_found = True
        else:
            dbp_found = True

    for doc in res2['hits']['hits']:
        uri = doc['_source']['uri']
        if uri not in props.keys() :
            props[uri] = {'label':doc['_source']['label'],'domain':doc['_source']['domain'],'range':doc['_source']['range'],'synonyms':doc['_source']['synonyms']}
            uniq_labels.append(uri.split("/")[-1])

        prop_type=uri.split("/")[-2]

        if prop_type=="ontology":
            dbo_found = True
        else:
            dbp_found = True


    if dbp_found and not dbo_found:
        # print("Here Here")
        for label in uniq_labels:
            # print("Searched : ",dbo_prefix + label)
            query ={"size": 10,"query": {"wildcard": {"uri": "*"+label}}}
            res = es.search(index="properties", doc_type="dbo-relation", body=query, request_timeout=20)

            for doc in res['hits']['hits']:
                uri = doc['_source']['uri']
                if uri not in props.keys():
                    props[uri] = {'label': doc['_source']['label'], 'domain': doc['_source']['domain'],
                                  'range': doc['_source']['range'], 'synonyms': doc['_source']['synonyms']}

    elif dbo_found and not dbp_found:
        # print("Here Here")
        for label in uniq_labels:
            # print("Searched : ",dbo_prefix + label)
            query ={"query": {"wildcard": {"uri": "*"+label}}}
            res = es.search(index="properties", doc_type="dbo-relation", body=query, request_timeout=20)

            for doc in res['hits']['hits']:
                uri = doc['_source']['uri']
                if uri not in props.keys():
                    props[uri] = {'label': doc['_source']['label'], 'domain': doc['_source']['domain'],
                                  'range': doc['_source']['range'], 'synonyms': doc['_source']['synonyms']}
    return props


def search_triple(tstrs):
    triples = []

    '''
        so we search by all combined by "_" if miss, search by each 
        but results fileter on existance of full word on splits by _ or space [From the seacrh module]
    '''

    for tstr in tstrs.split(" "):
        query1 = {"size": 10,"query": {"bool":{"must": [ {"wildcard": {"doc.sphrase.keyword" : "*"+tstr+"*"}}]}}}

        # , "should": [
        query2 = {"size": 10,"query": {"bool":{"must": [ {"wildcard": {"doc.ophrase.keyword" : "*"+tstr+"*"}}]}}}

        res1 = es.search(index="dbpedia_triples", doc_type="triple",body=query1,request_timeout=20)
        res2 = es.search(index="dbpedia_triples", doc_type="triple",body=query2,request_timeout=20)

        for doc in res1['hits']['hits']:
            e_label_list = doc['_source']['doc']['subject'].split("/")[-1].split("_")
            if doc['_source']['doc']['predicate'].strip().find('wikiPage')==-1 and tstr in e_label_list :
                # print("  %s       %s       %s      %s  " % (doc['_id'],doc['_source']['doc']['subject'],doc['_source']['doc']['predicate'],doc['_source']['doc']['object'])) #doc['_source']['subject'],
                triples.append([doc['_id'],doc['_source']['doc']['subject'],doc['_source']['doc']['predicate'],doc['_source']['doc']['object']])

        for doc in res2['hits']['hits']:
            e_label_list = doc['_source']['doc']['subject'].split("/")[-1].split("_")
            if doc['_source']['doc']['predicate'].strip().find('wikiPage')==-1 and tstr in e_label_list:
                # print("  %s       %s       %s      %s  " % (doc['_id'],doc['_source']['doc']['subject'],doc['_source']['doc']['predicate'],doc['_source']['doc']['object'])) #doc['_source']['subject'],
                triples.append([doc['_id'],doc['_source']['doc']['subject'],doc['_source']['doc']['predicate'],doc['_source']['doc']['object']])


    return triples

def abstracts_to_es():

    abs_files = os.listdir(abstarcts_folder)
    i=0
    for abs_f in abs_files[0:2]  :
        abs_triples = []
        with open(abstarcts_folder+"/"+abs_f, 'r') as f: #Add os.pathseperator
            rows = csv.reader(f)
            k=1
            for row in rows:
                if row[0].strip()[0:4].strip() == 'http':
                    abs_triples.append(row)

                    doc = {}

                    if abs_triples[-1][-1].find('*')>-1:
                        # print("\n line " + abs_triples[-1][-1])

                        doc['uri'] = row[1]
                        doc['abstract'] = row[2]

                        res = es.index(index="dbpedia_abstracts", doc_type='abstract', id=k, body=doc)
                else:
                    abs_triples[-1] = abs_triples[-1]+" "+' '.join(part.stip() for part in row)

                    print("\n The K : "+str(k)+" line "+abs_triples[-1])
                    print("\n")

                k = k+1

            f.close()
        del abs_triples
        gc.collect()

        i = i+1

        if i>2:
            break

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

def index_triples_on_es():
    dbp_files_folder = ""
    triples_files = os.listdir(dbp_files_folder)

    docs = {}
    k = 1

    try:
        for tf in triples_files:
            if k<1298317 :
                k = k+1
                continue
            with open(dbp_files_folder+"/"+tf, 'r') as f: #Add os.pathseperator
                rows = csv.reader(f)
                doc = {}
                for row in rows :
                    doc['subject'] = row[1]
                    doc['predicate'] = row[0]
                    doc['object'] = row[2]

                    k = k + 1

                    if len(doc) % 500 == 0:
                        res = es.index(index="dbpedia_triples", body=docs)
                    docs['triple'] = doc

    except Exception as ex:
        with open("dbpedia/missed.txt", 'a') as log:
            log.write("Error on prop "+k+" : "+ex)
        print('Exception : ', ex)

def entitySearch(query):
    indexName = ["wikiindexlabels","wikiindexaltlabels"]
    docType = "document"
    entities = []
    elasticResults = es.search(index=indexName, doc_type=docType, body={"query": {

            "bool": {
                "must": {
                    "bool": {"should": [
                        {"multi_match": {"query": query.strip(),
                                         "fields": ["doc.knownAs","doc.label"], "fuzziness": 6}},
                        {"multi_match": {"query": query.strip(), "fields": ["doc.knownAs","doc.label"]}},
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

    # print("Entities : ", entities)

    return entities

def ontologySearch(query):
    indexName = "dbontologyindex"
    docType = "document" # Change this to use it
    results = []
    elasticResults = es.search(index=indexName, doc_type=docType, body={
        "query": {
            "bool": {
                "must": {
                    "bool": {"should": [
                        {"multi_match": {"query": "http://dbpedia.org/ontology/" + query.replace(" ", ""),
                                         "fields": ["uri"], "fuzziness": "AUTO"}},
                        {"multi_match": {"query": query, "fields": ["label"]}},
                    ]}
                }
            }
        }
        , "size": 15
    })
    # print(elasticResults)
    for result in elasticResults['hits']['hits']:
        if not result["_source"]["uri"][result["_source"]["uri"].rfind('/') + 1:].istitle():
            results.append((result["_source"]["label"], result["_source"]["uri"]))
    return results
    # for result in results['hits']['hits']:
    # print (result["_score"])
    # print (result["_source"])
    # print("-----------")


def propertySearch(query):
    indexName = "properties"
    results = []
    elasticResults = es.search(index=indexName, doc_type='dbo-relation', body={
        "query": {
            "bool": {
                "must": {
                    "bool": {"must": [
                        {"multi_match": {"query": query, "fields": ["label","synonyms"]}},
                        {"multi_match": {"query": "http://dbpedia.org/property/" + query.replace(" ", ""),
                                         "fields": ["uri"], "fuzziness": "AUTO"}}]}
                }
            }
        }
        , "size": 20})
    for result in elasticResults['hits']['hits']:
        results.append((result["_source"]["label"], result["_source"]["uri"]))
    return results



def evaluateWiki():
    qs_file = os.path.abspath(os.path.join("/home/vyas/dataset/", "entityData_Sep/surfaceForm_wikiEntity_0_25.csv"))
    all_questions = []
    with open(qs_file, "r", encoding='utf-8') as qf:
        qf_loaded = csv.reader(qf)
        for i,line in enumerate(qf_loaded):
            all_questions.append([line[0].strip(),line[1].strip()])
    qf.close()

    correct_count = 0
    eval_res = []

    for line in all_questions[1:]:
        res = entitySearch(line[0].strip())

        if len(res)>0:
            line.extend([val[1] for val in res[:3]])
            eval_res.append(line)

        with open("/home/vyas/dataset/entityData_Sep/elasticsearch.csv", "ab+") as f:
            wr = ",".join(line)+"\n"
            f.write(wr.encode('utf-8'))
        f.close()

#This function simply compares the Levenstein distances
def lev_similarity(query, field):

    numerator = 20
    lenth = len(query)

    dist_stems = editdistance.eval(query, field)
    denominator = float((dist_stems+1) * lenth)

    return (numerator/denominator)




if __name__ == '__main__':
    # index_on_es()
    # process_abstracts()
    # abstracts_to_es()
    # get_AllKBproperties()
    # index_triples_on_es()
    
    #bulk(es, index_bulk_labels())
    #bulk(es, index_bulk_knownAs())
    
    # search_es("dbpedia_props","http://dbpedia.org/ontology/discoverer") #http://dbpedia.org/property/discoverer
    # search_es("dbpedia_props", "dbp-relation","author")
    # search_es("dbpedia_triples","triple","Johnny_Carson")
    # index_bulk_triples_es()
    # print("Returned", search_triple('Obama'))
    # search_triple("Obama")
    # print(search_prop('wife','wife'))

    # print(propertySearch("wife"))
    # index_on_es()
    # print(propertySearch("paint"))

    print("Found Entities : ",entitySearch('nepal'))
    
    #evaluateWiki()
