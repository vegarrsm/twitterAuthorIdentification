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

api = tweepy.API(auth)
ans = "y"
dataSize = 10 #Amount of tweets to fetch (this includes testing tweets, so add 10% compared to desired training set size)
outputFileName = "10TweetResults.csv"
results = []
inputList = [["@BarackObama", "@katyperry","@justinbieber"], ["@rihanna", "@taylorswift13","@Cristiano"], ["@ladygaga","@realDonaldTrump","@TheEllenShow"], ["@ArianaGrande", "@YouTube", "@KimKardashian"], ["@jtimberlake", "@selenagomez", "@Twitter"], ["@narendramodi", "@cnnbrk", "@britneyspears"],["@ddlovato", "@shakira", "@jimmyfallon"], ["@BillGates", "@CNN", "@neymarjr"]]
##Code for inputing twitter users to console. Decided to use prepared list for experiment
# user = "s"
# while user != "":#Add validation of screen name
#     user = input("\ninput twitter screen name of user (remember '@'): ")
#     if user != "":
#         users.append(user)
#     print("leave empty when finished")

for users in inputList:
    start_time = time.time()

    f = open("authors.txt","w", encoding="utf-8")
    t = open("testInputFile.txt", "w", encoding="utf-8")
    f.write("".join(users))
    p = re.compile(".*(?=http)")
    for user in users:
        amount = 0
        f.write("\n\nseparation---NEW USER---separation\n" + user + "\n\n")
        t.write("\n\nseparation---NEW USER---separation\n" + user + "\n\n")
        for status in tweepy.Cursor(api.user_timeline, screen_name=user, tweet_mode="extended", count = 200).items(): #Bruk api.user_timeline(...) istedet for å få count og full_text til å funke?
            m = p.match(status.full_text)
            if m and m.group()[:2] != "RT":
                if amount <= dataSize - dataSize//10:
                    f.write(m.group() + "\n\n")
                else:
                    t.write(m.group() + "\n\n")
                if amount >= dataSize:
                    break
                amount += 1
        f.write("\n\nseparation---NEW USER---separation")
        t.write("\n\nseparation---NEW USER---separation")
        print("Amount for " + user[1:] + ": ", amount)
    f.close()
    t.close()
    print("--- %s seconds ---" % (time.time() - start_time),"\n")

    
    testRes = []
    i = -1
    for user in users:
        t = open("testInputFile.txt", "r+", encoding="utf-8")
        text = re.sub(r"separation---NEW USER---separation\n@testUser", "separation---NEW USER---separation\n" +users[i], t.read()) if i >= 0 else t.read()
        text = re.sub(r"separation---NEW USER---separation\n" + user, "separation---NEW USER---separation\n@testUser" ,text)
        t.seek(0)
        t.truncate()
        t.write(text)
        t.close()
        scriptRes = script.main()
        print("result for: " + user, scriptRes)
        testRes += scriptRes + [1 if user[1:] == scriptRes[0] else 0]
        i += 1
    t.close()
    results.append(testRes)

print("\n\n")


f = open(outputFileName, 'w')
with f:
    writer = csv.writer(f)
    for row in results:
        writer.writerow(row)