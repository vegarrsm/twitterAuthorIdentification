import tweepy
import re
import time
import os
import script
import csv
import pandas as pd

f = pd.read_csv("authors.tsv", delimiter='\t', header=None, names=['ID', 'author', 'NA', 'sentence'])

out_file = open("authors.tsv","wt", encoding="utf-8")
write = csv.writer(out_file, delimiter='\t')

i = 0
for index,x in f.iterrows():
    write.writerow([index,x["author"], x["NA"],x["sentence"]])
    i = index

f = pd.read_csv("testInputFile.tsv", delimiter='\t', header=None, names=['ID', 'author', 'NA', 'sentence'])

out_file = open("testInputFile.tsv","wt", encoding="utf-8")
write = csv.writer(out_file, delimiter='\t')

for index,x in f.iterrows():
    write.writerow([index+i+1,x["author"], x["NA"],x["sentence"]])