#!/bin/python2.7

import pandas as pd
import numpy as np
import random
import cPickle as pickle
import gensim.utils
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import glob
import os.path
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

trainFold='Intermediate/Training_Lists/'
valFold = 'Intermediate/Validation_Lists/'
dfFold = 'DFs_TopicProbs/'

#loading selected lda model
lda=LdaModel.load('LDA_Models/full_lda_60', mmap=None)

#for each of the three revDF files, split into main and validation sets
for revDF in glob.glob('Intermediate/Cleaned_DFs/review*cleaned*'):
    dfName=revDF.split('/')[2].split('_')[0]
    trainDF=pd.io.pickle.read_pickle(revDF)

    with open(valFold+'valIndices'+dfName+'.pickle', 'rb') as f:
        valIndices=pickle.load(f)
    valIndices.sort()


    #extract validation observations into separate df and pickle
    mask = map(lambda x: x in valIndices, trainDF.index)
    valDF=trainDF[mask]
    with open(dfFold+'val_'+dfName+'.pickle', 'w+') as f:
        pickle.dump(valDF, f)
    valDF=0

    #remove validation observations from trainDF
    trainDF.drop(valIndices)

    #load list of stripped training reviews, and find topic probabilities associated with each review
    revList=pd.io.pickle.read_pickle(trainFold+dfName+'list.pickle')
    dicionary=gensim.corpora.Dictionary.load('Intermediate/LDA_Overhead/dictionary.dict')
    corpus=[dictionary.doc2bow(review) for review in revList]
    topicProbs=lda.inference(corpus)
    lda=0

    #initialize (number of reviews)x60 df containing topic probabilities
    topicNames=np.arange(60)
    probDF=pd.DataFrame(columns=topicNames,index=trainDF.index)

    #fill in columns with topic probabilities
    for ind,rev in enumerate(topicProbs[0]):
        for topic, prob in enumerate(rev):
            probDF=probDF.set_value(ind, topic, prob)

    #add probDF columns to trainDF
    trainDF=pd.concat([trainDF,probDF], axis=1)

    #pickle trainDF
    with open(dfFold+dfName+'.pickle', 'w+') as f:
        pickle.dump(trainDF, f)
    trainDF=0

