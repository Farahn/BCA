# takes the json file reviews.json, and takes the review text and stars and stores as two csv files, training and test
# creates folder data if it is not in the directory. 

import pandas as pd
import numpy as np
import csv
import json
import os

def get_review_data(path,cols, csv_path):
    
    w = csv.writer(open(csv_path,'w'))
    w.writerow(cols)
    with open(path) as data_file:
        for line in data_file:
            data = json.loads(line)
            w.writerow([data[k] for k in cols])

directory = 'data'
if not os.path.exists(directory):
    os.makedirs(directory)
            
cols = ['text','stars']
get_review_data('review.json', cols, 'data/review_text_all.csv')

df = pd.read_csv('/s0/farahn/dataset/review_text.csv')
df = df.sample(frac = 1.0)

test = 0.2

df_test, df_train = df[:int(len(df)*test)], df[int(len(df)*(test)):]

df_test[['text','stars']].to_csv('data/test.csv')
df_train[['text','stars']].to_csv('data/review_text.csv')
