import pandas as pd
import numpy as np
import pickle
import glob


def SplitPriceLevel(in_folder):
    for df_file in glob.glob(in_folder+'*'):

        state = df_file.split('/')
        state = state[len(state)-1].split('.')[0]

        df = pd.io.pickle.read_pickle(df_file)

        df_high = df[df['attributes_Price Range']== 3]
        df_low = df[df['attributes_Price Range']== 2]

        with open(in_folder+state+'_high.pickle', 'w+') as f:
            pickle.dump(df_high, f)

        with open(in_folder+state+'_low.pickle', 'w+') as f:
            pickle.dump(df_low, f)



