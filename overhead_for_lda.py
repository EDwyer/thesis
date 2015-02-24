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

# set up for stemming words
stemmer=SnowballStemmer('english')

exclude = ['"', "'", '.', '!', '?', ','] #list of punctuation to remove

def strip_func(rev): #function stems words and removes punctuation & common stopwords
    rev = re.sub(r'[^\w]', ' ', rev)
    rev = rev.split(' ')
    rev = [w for w in rev if not w in stopwords.words('english')]
    rev = [w for w in rev if not w in stopwords.words('english')]
    rev = [w for w in rev if not w.isdigit()]
    rev = map(stemmer.stem, rev)
    return ' '.join(rev)

revListFile = 'Intermediate/LDA_Overhead/revList.pickle'
valListFile = 'Intermediate/LDA_Overhead/valList.pickle'
revList=list()
valList=list()

# read in the review data frames from pickle, divide each df into training and validation sets and save to txt files
for revDF in glob.glob('Intermediate/Cleaned_DFs/review*cleaned*'):
    if not len(glob.glob('Intermediate/Stripped_Validation_Rev_TXTs/*'))==3:
        revList = pd.io.pickle.read_pickle(revDF)
        revList = revList.text.tolist()
        revList = [rev for rev in revList if rev!='']

        # but first, remove 20% of the reviews from revList and put into valList to use as a validation set
        length = len(revList)
        twentyPercent= int(length*0.20)
        valIndices=random.sample(xrange(0,length), twentyPercent)
        valIndices.sort(reverse=True)
        for index in valIndices:
            valList.append(revList[index])
            del revList[index]

        #defining filepaths to save .txt files of sorted reviews:
        training_original='Intermediate/Unstripped_Training_Rev_TXTs/'+revDF.split('/')[2].split('_')[0]+'_training_original.txt'
        validation_original='Intermediate/Unstripped_Validation_Rev_TXTs/'+revDF.split('/')[2].split('_')[0]+'_validation_original.txt'
        training_stripped='Intermediate/Stripped_Training_Rev_TXTs/'+revDF.split('/')[2].split('_')[0]+'_training_stripped.txt'
        validation_stripped='Intermediate/Stripped_Validation_Rev_TXTs/'+revDF.split('/')[2].split('_')[0]+'_validation_stripped.txt'

        # printing the unedited training set reviews to a .txt file
        if not os.path.isfile(training_original):
            with open(training_original, 'w+') as file:
                for item in revList:
                    print>>file, item

        #stripping the training set and printing these stripped reviews to a .txt. file
        if not os.path.isfile(training_stripped):
            revList = map(strip_func, revList)
            with open(training_stripped, 'w+') as file:
                for item in revList:
                    print>>file, item

        #printing the unedited validation set reviews to a .txt file
        if not os.path.isfile(validation_original):
            with open(validation_original, 'w+') as file:
                for item in valList:
                    print>>file, item

        #stripping the validation set and printing these stripped reviews to a .txt. file
        if not os.path.isfile(validation_stripped):
            validList = map(strip_func, valList)
            with open(validation_stripped, 'w+') as file:
                for item in validList:
                    print>>file, item

        #clearing individual revList and valList
        revList=list()
        valList=list()

# combining the stripped training set .txt files into a pickled list of lists named revList
if not os.path.isfile(revListFile):
    for revTxt in glob.glob('Intermediate/Stripped_Training_Rev_TXTs/*'):
        f = open(revTxt, 'r')
        curr = f.read()
        f.close()
        curr = curr.split('\n')
        curr = map(lambda x: x.split(' '), curr)
        revList = revList + curr
    with open(revListFile, 'w+') as f:
        pickle.dump(revList,f)
    revList=list()

# combining the stripped validation set .txt files into a pickled list of lists named valList
if not os.path.isfile(valListFile):
    for revTxt in glob.glob('Intermediate/Stripped_Validation_Rev_TXTs/*'):
        f = open(revTxt, 'r')
        curr = f.read()
        f.close()
        curr = curr.split('\n')
        curr = map(lambda x: x.split(' '), curr)
        valList = valList + curr
    with open(valListFile, 'w+') as f:
        pickle.dump(valList,f)
    valList=list()

#creating a dictionary from RevList
dictFile='Intermediate/LDA_Overhead/dictionary.dict'
if not os.path.isfile(dictFile):
    with open(revListFile, 'rb') as f:
        revList=pickle.load(f)
    dictionary = gensim.corpora.Dictionary(revList)
    # removing the top and bottom 5% most frequent words
    dictLen=len(dictionary.keys())
    no_below=int(0.05*dictLen)
    no_above=0.05
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    dictionary.save(dictFile)
    dictionary=0

#creating training set corpus
corpusFile='Intermediate/LDA_Overhead/corpus.mm'
if not os.path.isfile(corpusFile):
    with open(revListFile, 'rb') as f:
        revList=pickle.load(f)
    dictionary=gensim.corpora.Dictionary.load('Intermediate/LDA_Overhead/dictionary.dict')
    corpus = [dictionary.doc2bow(review) for review in revList]
    MmCorpus.serialize(corpusFile, corpus)
    corpus=list()

#creating validation corpus
valCorpusFile='Intermediate/LDA_Overhead/valCorpus.mm'
if not os.path.isfile(valCorpusFile):
    with open(valListFile, 'rb') as f:
        valList=pickle.load(f)
    dictionary=gensim.corpora.Dictionary.load('Intermediate/LDA_Overhead/dictionary.dict')
    valCorpus = [dictionary.doc2bow(review) for review in revList]
    MmCorpus.serialize(valCorpusFile, valCorpus)
    valCorpus=list()

