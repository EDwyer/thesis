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

def strip_func(rev): #function removes short words(< 3), stems words, and removes punctuation & common stopwords
    rev = re.sub(r'[^\w]', ' ', rev)
    rev = rev.split(' ')
    rev = [w for w in rev if not w in stopwords.words('english')]
    rev = [w for w in rev if len(w)>2]
    rev = [w for w in rev if not w.isdigit()]
    rev = map(stemmer.stem, rev)
    return rev

valListFile = 'Intermediate/LDA_Overhead/valList.pickle'
trainFold = 'Intermediate/Training_Lists/'
valFold = 'Intermediate/Validation_Lists/'


# read in the review data frames from pickle, divide each df into training and validation sets and re-pickle
for revDF in list(['Intermediate/Cleaned_DFs/reviewDF1_cleaned']):
#for revDF in glob.glob('Intermediate/Cleaned_DFs/review*cleaned*'):
    dfNumber= revDF.split('/')[2].split('_')[0]

    if not len(glob.glob(valFold+'rev*'))==6:
        print('beginning '+dfNumber)
        trainList = pd.io.pickle.read_pickle(revDF)
        trainList = trainList.text.tolist()
        trainList = [rev for rev in trainList if rev!='']

        # remove 20% of the reviews from trainList and put into valList to use as a validation set
        length = len(trainList)
        twentyPercent= int(length*0.20)
        valList=list()
        valIndices=random.sample(xrange(0,length), twentyPercent)
        with open(valFold+'valIndices'+dfNumber+'.pickle', 'w+') as f:
            pickle.dump(valIndices, f)
        valIndices.sort(reverse=True)
        for index in valIndices:
            valList.append(trainList[index])
            del trainList[index]

        #defining filepaths to save .pickle files of sorted reviews:
        trainFile=trainFold+dfNumber+'list.pickle'
        valFile=valFold+dfNumber+'list.pickle'

        #stripping the training set and pickling these stripped reviews
        if not os.path.isfile(trainFile):
            trainList = map(strip_func, trainList)
            with open(trainFile, 'w+') as f:
                pickle.dump(trainList, f)
            print('trainList'+dfNumber+'pickled')
        trainList=list()
        #stripping the validation set and pickling these stripped reviews
        if not os.path.isfile(valFile):
            valList = map(strip_func, valList)
            with open(valFile, 'w+') as f:
                pickle.dump(valList, f)
            print('valList'+dfNumber+'pickled')
        valList=list()

# combining the stripped training set files into a pickled list of reviews named trainList
trainList=list()
trainFile=trainFold+'trainList.pickle'
if not os.path.isfile(trainFile):
    for revPkl in glob.glob(trainFold+'*'):
        with open(revPkl, 'rb') as f:
            currList=pickle.load(f)
        trainList = trainList + currList
    with open(trainFile, 'w+') as f:
        pickle.dump(trainList,f)
    print('trainList pickled')
else:
    with open(trainFile, 'rb') as f:
        trainList=pickle.load(f)

#creating a dictionary from trainList, using only the 10,000 most frequent words
dictFile='Intermediate/LDA_Overhead/dictionary.dict'
if not os.path.isfile(dictFile):
    dictionary = gensim.corpora.Dictionary(trainList, prune_at=10000)
    dictionary.save(dictFile)
else:
    dictionary=gensim.corpora.Dictionary.load(dictFile)

#creating training set corpus
corpusFile='Intermediate/LDA_Overhead/corpus.mm'
if not os.path.isfile(corpusFile):
    corpus = [dictionary.doc2bow(review) for review in trainList]
    MmCorpus.serialize(corpusFile, corpus)
    corpus=0
    print('corpus pickled')

#zeroing-out trainList for memory concerns
trainList=0

# combining the stripped validation set files into a pickled list of reviews named valList
valList=list()
valFile=valFold+'valList.pickle'
if not os.path.isfile(valFile):
    for revPkl in glob.glob(valFold+'rev*'):
        with open(revPkl, 'rb') as f:
            currList=pickle.load(f)
        valList = valList + currList
    with open(valFile, 'w+') as f:
        pickle.dump(valList,f)
    print('valList pickled')
else:
    with open(valFile, 'rb') as f:
        valList=pickle.load(f)


#creating validation corpus
valCorpusFile='Intermediate/LDA_Overhead/valCorpus.mm'
if not os.path.isfile(valCorpusFile):
    valCorpus = [dictionary.doc2bow(review) for review in valList]
    MmCorpus.serialize(valCorpusFile, valCorpus)
    valCorpus=0
    print('valCorpus pickled')
else: valList=0
