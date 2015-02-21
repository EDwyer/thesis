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

## set up for stemming words
#stemmer=SnowballStemmer('english')
#
#exclude = ['"', "'", '.', '!', '?', ','] #list of punctuation to remove
#
#def strip_func(rev): #function stems words and removes punctuation & common stopwords
#    rev = re.sub(r'[^\w]', ' ', rev)
#    rev = rev.split(' ')
#    rev = [w for w in rev if not w in stopwords.words('english')]
#    rev = [w for w in rev if w !='']
#    rev = map(stemmer.stem, rev)
#    return ' '.join(rev)
#
#revListFile = 'Intermediate/LDA_Overhead/revList.txt'
#if not os.path.isfile(revListFile):
#    # reading in the review data frames from pickle, applying strip_func, and saving to txt
#    for revDF in glob.glob('Intermediate/Cleaned_DFs/review*cleaned*'):
#        revList = pd.io.pickle.read_pickle(revDF)
#        revList = revList.text.tolist()
#        revList = [rev for rev in revList if rev!='']
#
#        # but first, remove 20% of the reviews to use as a validation set
#        validList = list()
#        length =len(revList)
#        twentyPercent=int(length)-100
#        #twentyPercent=int(length*0.20)
#        validationIndexes=random.sample(xrange(0,length), twentyPercent)
#        validationIndexes.sort(reverse=True)
#        for index in validationIndexes:
#            validList.append(revList[index])
#            del revList[index]
#
#        # printing the undedited training set reviews to a .txt file
#        textFile = 'Intermediate/Unstripped_Training_Rev_TXTs/'+revDF.split('_')[0]+'_training_original.txt'
#        with open(textFile, 'w') as file:
#            for item in revList:
#                print>>file, item
#
#        #printing the unedited validation set reviews to a .txt file
#        textFile = 'Intermediate/Unstripped_Validation_Rev_TXTs/'+revDF.split('_')[0]+'_validation_original.txt'
#        with open(textFile, 'w') as file:
#            for item in validList:
#                print>>file, item
#
#        #stripping the training set and printing these stripped reviews to a .txt. file
#        revList = map(strip_func, revList)
#
#        textFile_stripped = 'Intermediate/Stripped_Training_Rev_TXTs/'+revDF.split('_')[0]+'_training_stripped.txt'
#        with open(textFile_stripped, 'w') as file:
#            for item in revList:
#                print>>file, item
#
#        #stripping the validation set and printing these stripped reviews to a .txt. file
#        validList = map(strip_func, validList)
#
#        textFile_stripped = 'Intermediate/Stripped_Validation_Rev_TXTs/'+revDF.split('_')[0]+'_validation_stripped.txt'
#        with open(textFile_stripped, 'w') as file:
#            for item in validList:
#                print>>file, item
#
#
#
#    # combining the stripped txt files into a corpus named revList
revList = list()

for revTxt in glob.glob('Intermediate/Stripped_Training_Rev_TXTs/review*training_stripped.txt'):
    f = open(revTxt, 'r')
    curr = f.read()
    f.close()
    curr = curr.split('\n')
    curr = map(lambda x: x.split(' '), curr)
    revList = revList + curr



#creating a dictionary from RevList
dictionary = gensim.corpora.Dictionary(revList)
# removing the top and bottom 5% most frequent words
dictLen=len(dictionary.keys())
no_below=int(0.05*dictLen)
dictionary.filter_extremes(no_below=no_below, no_above=0.05, keep_n=None)
dictionary.save('Intermediate/LDA_Overhead/dictionary.dict')

#if not os.path.isfile(corpusFile):
#creating corpus
corpus = [dictionary.doc2bow(review) for review in revList]
corpusFile = 'Intermediate/LDA_Overhead/corpus.mm'

#initializing lda
dictionary=0
corpus=0
lda=0
for i in list([4, 8, 15, 20, 25, 40, 50, 60 ]):
    dictionary=gensim.corpora.Dictionary.load('Intermediate/LDA_Overhead/dictionary.dict')
    corpus=gensim.corpora.MmCorpus('Intermediate/LDA_Overhead/corpus.mm')
    lda=LdaModel(corpus=corpus, num_topics=i, id2word=dictionary)
    fileName='LDA_Models/full_lda_'+str(i)
    lda.save(fileName)
    lda=0
    corpus=0
    dictionary=0

