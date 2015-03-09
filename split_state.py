import pandas as pd
import numpy as np
import cPickle as pickle
import gensim.utils
import glob
import os

in_file = 'Intermediate/all_reviews/review*'
out_folder = 'Intermediate/all_reviews_states/'


#sorting cleaned review dataframes (augmented with business data) into one file for each city/state (conveniently, there is only one metropolitan area per state)

for rev_df in glob.glob(in_file+'*'):
    df_number=rev_df.split('/')[2].split('_')[0].split('F')[1]
    print('sorting review_df_cleaned_' + df_number)
    rev_list = pd.io.pickle.read_pickle(rev_df)

    by_state = rev_list.groupby('state')
    for state, state_revs in by_state:
	if state in ['WI', 'NV', 'AZ', 'PA', 'IL', 'NC']:
            state_file=out_folder+state+'.pickle'
            if not os.path.isfile(state_file):
                with open(state_file, 'w+') as f:
                    pickle.dump(state_revs, f)
            else:
	        old_file = pd.io.pickle.read_pickle(state_file)
                new_file = pd.concat([old_file, state_revs])
                with open(state_file, 'w+') as f:    
                     pickle.dump(new_file, f)

