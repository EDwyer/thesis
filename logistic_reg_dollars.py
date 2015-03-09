#!/bin/python2.7

import pandas as pd
import numpy as np
import random
import cPickle as pickle
import glob
import os.path
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score


# inputs:
    # in_file, the filepath of the dataframe with topic probabilities
    # one_vs: the number of $ to use as "one", compared to "all" other #s of $
    # non_topic_indpt=optional list of other (non categorical) parameters to regress on

def LogRegPriceLevel(in_file, one_vs, non_topic_indpt=[]):

    # get state from in_file and define outfiles

    state = in_file.split('.')[0].split('/')
    state = state[len(state)-1]

    # read in data from pickle
    state_df = pd.io.pickle.read_pickle(in_file)


    # drop unwanted columns and observations

    topic_names = map(lambda x: 'topic_{}'.format(x), np.arange(0,56))
    dependent = 'attributes_Price Range'

    to_keep = [dependent] + non_topic_indpt + topic_names

    state_df = state_df[to_keep]

    state_df.dropna(axis = 0, how = "any")


    # create dummy variable for price (1=one_vs, 0 = all) and drop Price Range

    state_df['one_vs'] = (state_df[dependent] == one_vs)
    state_df['one_vs'] = state_df['one_vs'].astype(int)
    state_df = state_df.drop(dependent , axis = 1)


    # prepare for logistic regression: use dmatrices to split into X and y

    var_names = 'one_vs ~ '
    topic_names = ''
    non_topic_names = ''

    for i in np.arange(0,56):
        topic_names = topic_names+'topic_'+str(i)+' + '
    topic_names = topic_names[:len(topic_names)-2]

    if non_topic_indpt != []:
        for name in non_topic_indpt:
            non_topic_names = non_topic_names + name + ' + '
        non_topic_names = non_topic_names[:len(non_topic_names)-2]

        var_names = var_names + topic_names + '+' + non_topic_names

    else:
        var_names = var_names + topic_names

    y, X = dmatrices(var_names, state_df, return_type = "dataframe")

    y = np.ravel(y)

    print 'X.shape = %s y.shape = %s' %(y.shape, X.shape)


#    # perform logistic regression and print output
#
#    model = LogisticRegression()
#    model =  model.fit(X,y)
#
#    print('R^2 = '+str(model.score(X,y)))
#    print('coefficients:')
#    print(model.coef_)


    #training set and test set division

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


    # initialize logistic regression, print output

    model = LogisticRegression()
    model.fit(X_train,y_train)

    print('R^2 = '+str(model.score(X,y)))
    print('coefficients:')
    print(model.coef_)

    with open(in_file.split('.')[0]+'_pl_coeffs.pickle', 'w+') as f:
        pickle.dump(model.coef_, f)

    # calculate and print accuracy of prediction

    predicted = model.predict(X_test)
    print('metrics.accuracy_score(y_test, predicted) = ' +str(metrics.accuracy_score(y_test, predicted)))
    print('metrics.roc_auc_score(y_test, predicted) = ' +str(metrics.roc_auc_score(y_test, predicted)))


    # 10-fold cross-validation
    scores = cross_val_score(LogisticRegression(), X, y, scoring = 'accuracy', cv = 10)
    print('10_fold_cross_val_score = '+str(scores))

# code adapted from www.bogotobogo.com/python/scikit-learn/scikit-learn_logistic_regression.php
