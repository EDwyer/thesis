import pandas as pd
import numpy as np

#importing review csv and saving it in 3 equally sized pickled dataframes
currDF=pd.io.parsers.read_csv('Intermediate/CSVs/review.csv')
currDF=reviewDF[:375153]
currDF.to_pickle('Intermediate/Uncleaned_DFs/reviewDF1')

currDF=pd.io.parsers.read_csv('Intermediate/Cleaned_DFs/review.csv')
currDF=reviewDF[375154:750306]
currDF.to_pickle('Intermediate/Uncleaned_DFs/reviewDF2')

currDF=pd.io.parsers.read_csv('Intermediate/Cleaned_DFs/review.csv')
currDF=reviewDF[750307:]
currDF.to_pickle('Intermediate/Uncleaned_DFs/reviewDF3')


#importing business csv and saving it as a pickled dataframe
currDF=pd.io.parsers.read_csv('Intermediate/Cleaned_DFs/business.csv')
currDF.to_pickle('Intermediate/Uncleaned_DFs/businessDF')


#importing user csv and saving it as a pickled dataframe
currDF=pd.io.parsers.read_csv('Intermediate/Cleaned_DFs/user.csv')
currDF.to_pickle('Intermediate/Uncleaned_DFs/userDF')


