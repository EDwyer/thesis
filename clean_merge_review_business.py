import pandas as pd
import numpy as np
import glob

#load business dataframe from pickle
DF=pd.io.pickle.read_pickle('Intermediate/Uncleaned_DFs/businessDF')

#reduce dataframe to observations whose 'categories' column contains 'restaurants'
DF=DF[DF['categories'].str.contains("R,e,s,t,a,u,r,a,n,t")]

#extract remaining business_id's from dataframe as list
id_list=DF.business_id.tolist()

#pickle reduced dataframe as 'businessDF_cleaned' after sorting and indexing by 'business_id' (and renaming 'stars' column to 'star_business')
DF.sort(['business_id'])
DF.set_index(['business_id'])
DF=DF.rename(columns={'stars':'stars_business'})
DF.to_pickle('businessDF_cleaned')

#load review dataframes from pickles, keep only the observations whose business_id's are contained within 'id_list', sort by business_id and pickle resulting dataframes
for revDF in glob.glob('Intermediate/Cleaned_DFs/review*:cleaned*'):
    DF=pd.io.pickle.read_pickle(revDF)
    DF=DF[DF['business_id'].isin(id_list)]
    DEF.sort(['business_id'])
    DF.to_pickle(revDF)

#reduce memory usage by setting DF=0, since it is no longer needed
DF=0

#load cleaned business dataframe from pickle
bus=pd.io.pickle.read_pickle('Intermediate/Cleaned_DFs/businessDF_cleaned')

# make list of all business dataframe column names, then remove unwanted column names
to_keep=bus.columns.tolist()
to_keep.remove('attributes_Accepts Insurance')
to_keep.remove('attributes_Hair Types Specialized In')
to_keep.remove('full_address')
to_keep.remove('hours_Friday')
to_keep.remove('hours_Monday')
to_keep.remove('hours_Saturday')
to_keep.remove('hours_Sunday')
to_keep.remove('hours_Thursday')
to_keep.remove('hours_Tuesday')
to_keep.remove('hours_Wednesday')
to_keep.remove('latitude')
to_keep.remove('longitude')
to_keep.remove('attributes_Wheelchair Accessible')
to_keep.remove('type')
to_keep.remove('attributes_Dogs Allowed')
to_keep.remove('attributes_Smoking')
to_keep.remove('attributes_By Appointment Only')
to_keep.remove('attributes_Dietary Restrictions')
to_keep.remove('attributes_Good For Kids')

# kept columns: 'attributes_Accepts Credit Cards', 'attributes_Ages Allowed', 'attributes_Alcohol', 'attributes_Ambience', 'attributes_Attire', 'attributes_BYOB', 'attributes_BYOB/Corkage', 'attributes_Caters', 'attributes_Coat Check', 'attributes_Corkage', 'attributes_Delivery', 'attributes_Drive-Thru', 'attributes_Good For', 'attributes_Good For Dancing', 'attributes_Good For Groups' 'attributes_Good for Kids',  'attributes_Happy Hour', 'attributes_Has TV', 'attributes_Music', 'attributes_Noise Level', 'attributes_Open 24 Hours', 'attributes_Order at Counter', 'attributes_Outdoor Seating', 'attributes_Parking', 'attributes_Payment Types', 'attributes_Price Range', 'attributes_Take-out', 'attributes_Takes Reservations', 'attributes_Waiter Service',  'attributes_Wi-Fi', 'business_id', 'categories', 'city',  'name', 'neighborhoods', 'open', 'review_count', 'stars', 'state',

#note: Good for Kids and Good For Kids are different categories, but the lower case for has more observations


#reduce business dataframe to keep only desired columns (those in 'to_keep')
bus=bus[to_keep]

#pickle dataframe
bus.to_pickle('Intermediate/Cleaned_DFs/businessDF_cleaned')

print('about to merge')

#join review data frames wih business data frame, using business_id as an index
for revDF in glob.glob('Intermediate/Cleaned_DFs/review*cleaned*'):
    rev=pd.io.pickle.read_pickle(revDF)
    rev=pd.merge(rev, bus, on='business_id', how='inner')
    rev.to_pickle(revDF)




