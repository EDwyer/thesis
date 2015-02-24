#!/bin/python2.7

import pandas as pd
import cPickle as pickle
import gensim.utils
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from gensim.models.ldamodel import LdaModel
import sys
reload(sys)
sys.setdefaultencoding('utf8')

revListFile='Intermediate/LDA_Overhead/revList.pickle'
dictFile='Intermediate/LDA_Overhead/dictionary.dict'
corpusFile = 'Intermediate/LDA_Overhead/corpus.mm'

with open(revListFile, 'rb') as f:
    revList=pickle.load(f)
dictionary=gensim.corpora.Dictionary.load(dictFile)
corpus=MmCorpus(corpusFile)
#initializing lda
for i in list([10,12,14,16,18,20]):
    lda=LdaModel(corpus=corpus, num_topics=i, id2word=dictionary)
    fileName='LDA_Models/full_lda_'+str(i)
    lda.save(fileName)
    lda=0

