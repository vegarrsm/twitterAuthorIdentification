import sys
from Author import Author
import time
from math import log, exp
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import  GaussianNB
import seaborn

arguments = sys.argv

##Calculates posterior probability
def performNaiveBayes(authors, testAuthor):
    ##Converting values to log space as python will round multiplication to 0 as the 
    ##float value is too small. Will use logarithmic values during rest of calculation for this reason.
    for author in authors:
        prior = log(author.getPrior())
        conditional = 0
        for testWord in testAuthor.getWordProbability():
            conditional += log(author.getConditional(testWord))
        # calculating total probability here
        total = 0
        for a in authors:
            termTotal = 0
            for word in testAuthor.getWordProbability():
                termTotal += log(a.getConditional(word))
            total += termTotal + log(author.getPrior())
            if total == 0:
                print("Something is slightly wrong")
                total = log(0.01)
        author.setPosterior((prior + conditional)-total)
    return authors


# Used for printing the final results of the naive bayes
def results(authors):
    sortedA = sorted(authors, key=lambda author: author.getPosterior(), reverse=True)
    summed = sum(a.getPosterior() for a in sortedA)
    retList = []
    for author in sortedA:
        author.setPosterior(author.getPosterior()/summed)
        retList.append(author.getID())
        retList.append(author.getPosterior())
    return retList

# Entry of the script
def main():
    try:
        f = pd.read_csv("authors.tsv", delimiter='\t', header=None, names=['ID', 'author', 'NA', 'sentence'])
    except Exception as error:
        print(error)
        sys.exit()

    authors = list(set([x["author"][1:] for index, x in f.iterrows()]))
    for i in range(len(authors)):
        authors[i] = Author(authors[i],"authors.tsv")
    # Sets prior probability for each author
    for author in authors:
        author.setPrior(1.00/len(authors))
    return authors


#added for quicker testing of script.py, by not having to run authorSelect
if __name__ == "__main__":
    #Training cpu
    trainTime = time.time()
    authors = main()
    trainTime = time.time() - trainTime
    
    #Validation
    try:
        t = pd.read_csv("testInputFile.tsv", delimiter='\t', header=None, names=['ID', 'author', 'NA', 'sentence'])
    except Exception as error:
        print(error)
        sys.exit()

    authorNums = {}
    for i in range(len(authors)):
        authorNums[authors[i].getID()] = i 

    resSingleTweet = []
    trueList = []
    validationTime = time.time()
    for index, X in t.iterrows():
        testAuthor = Author(X["author"], "testInputFile.tsv", index)
        resSingleTweet.append([X["author"],results(performNaiveBayes(authors, testAuthor))])
        trueList.append(authorNums.get(X["author"][1:]))
    validationTime = time.time() - validationTime
    
    
    pred = []
    correct = []  
        
    for X in resSingleTweet:
        pred.append(1 if X[0][1:] == X[1][0] else 0)

    confusion = confusion_matrix(trueList, pred, [i for i in range(len(authors))])
    
    print("Average result single tweet: ", sum(pred)/len(pred))
    print("Validation time of all tweets: ", validationTime)
    print("Training time of all tweets: ", trainTime)

    labels = [auth[1] for auth in authorNums]
    seaborn.heatmap(confusion, annot=True, yticklabels = labels, xticklabels = labels)
