#!/bin/python2.7

import pandas as pd
import cPickle as pickle
import gensim.utils
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from gensim.models.ldamodel import LdaModel
import sys
reload(sys)
sys.setdefaultencoding('utf8')

dictFile='Intermediate/LDA_Overhead/dictionary.dict'
corpusFile = 'Intermediate/LDA_Overhead/corpus.mm'

dictionary=gensim.corpora.Dictionary.load(dictFile)
corpus=MmCorpus(corpusFile)
#initializing lda
for i in list([53, 54, 43]):
#for i in list([50, 20, 30, 40, 15, 45, 55, 60]):
    lda=LdaModel(corpus=corpus, num_topics=i, id2word=dictionary)
    fileName='LDA_Models/full_lda_'+str(i)
    lda.save(fileName)
    lda=0

