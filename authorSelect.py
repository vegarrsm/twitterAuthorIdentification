import tweepy
import re
import time
import os
import script
import csv

consumer_key = '2GGedBFBD4t323sjt76bWwUoy'
consumer_secret = 'qYRGDmtTkrKDuuMaeftv3TpdLb2BULi3rU1dMFtRq02lX3nhXi'
access_token = '1262678005484265472-z8ZmdbuOz5r4OqouAOPP87NcMvj9JG'
access_token_secret = 'vsJR8uQh9tDEio4qvxD6rHDWoeF94FjZw2FgI2JlZw4jb'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
dataSize = 1000 #Amount of tweets to fetch (this includes testing tweets, so add 10% compared to desired training set size)
results = []
#Default list set to biggest accounts on twitter
inputList = ["@BarackObama","@katyperry","@justinbieber","@rihanna","@BillGates", "@CNN", "@neymarjr", "@ArianaGrande", "@YouTube", "@KimKardashian", "@britneyspears","@ddlovato", "@shakira","@Cristiano"]#,"@jtimberlake", "@selenagomez","@narendramodi", "@cnnbrk"
##Code for inputing twitter users to console. Decided to use prepared list for experiment
#if input("Do you want to add custom list of twitter handles? input y for custom list or n for premade")=="y":
if False:
    dataSize = input("How many tweets should be fetched per user? ")
    inputList = []
    user = "temp"
    while user != "":
        user = input("\ninput twitter screen name of user (remember '@'): ")
        if user != "":
            inputList.append(user)
        print("leave empty when finished")

if(input("Refetch data? ") == "y"):
    ID = 0
    out_file_train = open("authors.tsv","wt", encoding="utf-8")
    BERTTrain = csv.writer(out_file_train, delimiter='\t')
    out_file_test = open("testInputFile.tsv","wt", encoding="utf-8")
    BERTTest = csv.writer(out_file_test, delimiter='\t')

    p = re.compile(".*(?=http)")
    for user in inputList:
        amount = 0
        for status in tweepy.Cursor(api.user_timeline, screen_name=user, tweet_mode="extended", count = 200).items(): #Bruk api.user_timeline(...) istedet for å få count og full_text til å funke?
            m = p.match(status.full_text)
            #Filters out retweets
            if m and m.group()[:2] != "RT" and m.group() != "":
                if amount <= dataSize - dataSize//10:
                    #Creating tsv training data
                    BERTTrain.writerow([ID, user, "a", m.group()])
                else:
                    #Creating tsv test/validation data
                    BERTTest.writerow([ID,user,"a", m.group()])
                if amount >= dataSize:
                    break
                amount += 1
                ID += 1
        print("Amount for " + user[1:] + ": ", amount)
    out_file_train.close()
    out_file_test.close()


i = -1
for user in inputList:
    scriptRes = script.main(user)
    results.append(1 if scriptRes[0] == user[1:] else 0)
    print("result for: " + user, scriptRes[0] == user[1:])
    print("time: ", scriptRes[-1])
    i += 1

print(results)
print("Average result: ", sum(results)/len(results))
