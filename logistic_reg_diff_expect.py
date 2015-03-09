#!/bin/python2.7

import pandas as pd
import numpy as np
import random
import cPickle as pickle
import glob
import os.path
import statsmodels.api as sm
import verhulst.stats as vs
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score


# inputs:
    # state_df, the state dataframe with added topic probabilities
    # non_topic_indpt=optional list of other (non categorical) parameters to regress on

# outputs:
    # coeffs: coefficients of logistic regression
    # hl_table: Hosmer-Lemeshow table


def LogRegDiffExpect(state_df, non_topic_indpt=[]):

    # drop unwanted columns and observations

    topic_names = map(lambda x: 'topic_{}'.format(x), np.arange(0,56))

    dependent = 'diff_expect'

    to_keep = [dependent] + non_topic_indpt + topic_names

    state_df = state_df[to_keep]

    state_df.dropna(axis = 0, how = "any")


    # prepare for logistic regression: use dmatrices to split into X and y

    var_names = 'diff_expect ~ '
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


    #training set and test set division

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print 'X_train'
    print X_train.shape
    print X_train
    print 'X_test'
    print X_test.shape
    print X_test
    print 'y_test'
    print y_test.shape
    print y_test
    print 'y_train'
    print y_train.shape
    print y_train

#    # initialize logistic regression, print output
#
#    model = LogisticRegression()
#    model.fit(X_train,y_train)
#
#    with open(out_folder+state+'_model.pickle', 'w+') as f:
#        pickle.dump(model, f)
#
#    coeffs = pd.DataFrame(model.coef_)
#    print('coeffs shape:')
#    print(coeffs.shape)


    # calculate accuracy of prediction

    predicted = model.predict_proba(X_test)[:,1]

    #hl = pd.DataFrame(vs.hosmer_lemeshow_table(y_test, predicted))

    #acc_score = metrics.accuracy_score(y_test, predicted)
    #roc_auc_score = metrics.roc_auc_score(y_test, predicted)

    #acc_scores = pd.Series([acc_score, roc_auc_score])

    # 10-fold cross-validation

    # cross_val_scores = cross_val_score(LogisticRegression(), X, y, scoring = 'accuracy', cv = 10)


    # return coefficients and test scores

    return(coeffs)


# initialize log reg

#for state in ['AZ', 'NV', 'WI']:
state = 'AZ'
   # print(state)
in_folder = 'logistic_reg/full_dfs/'
out_folder = 'logistic_reg/coeffs_testscores/'

df = pd.io.pickle.read_pickle(in_folder+state+'.pickle')
sample_rows = random.sample(df.index, 300)
df = df.ix[sample_rows]
df = df.groupby('attributes_Price Range')

coefficients = pd.DataFrame()
   # #hl_table = pd.DataFrame()
   # #acc_scores = pd.DataFrame()
   # #cross_val_scores = pd.DataFrame()

for item, group in df:
    print('current group key is %d' %item)
    (c)=LogRegDiffExpect(group)
    coefficients.append(c)
    #hl_table.append(hl)
    #acc_scores.append(a)
    #cross_val_scores.append(cv)

with open(out_folder+state+'_coefficients.pickle', 'w+') as f:
        pickle.dump(coefficients, f)

#with open(out_folder+state+'_hl_table.pickle', 'w+') as f:
#    pickle.dump(hl_table, f)
#with open(out_folder+state+'_acc_scores.pickle', 'w+') as f:
#    pickle.dump(acc_scores, f)
#with open(out_folder+state+'_cross_val.pickle', 'w+') as f:
#    pickle.dump(cross_val_scores, f)




# code adapted from www.bogotobogo.com/python/scikit-learn/scikit-learn_logistic_regression.php
