import sys
from Author import Author
import time
from math import log, exp
import pandas as pd

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


# Used for printing the final results of the naive bayes
def results(authors, start_time):
    sortedA = sorted(authors, key=lambda author: author.getPosterior(), reverse=True)
    summed = sum(a.getPosterior() for a in sortedA)
    retList = []
    for author in sortedA:
        author.setPosterior(author.getPosterior()/summed)
        retList.append(author.getID())
        retList.append(author.getPosterior())
    retList.append(time.time() - start_time)
    return retList

# Entry of the script
def main(validationAuthor):
    #testInput = input("Input testdata\n")
    start_time = time.time()
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
    # Creates a testAuthor instance of Author class just for the ease of further processing
    testAuthor = Author(validationAuthor[1:], "testInputFile.tsv")
    performNaiveBayes(authors, testAuthor)
    return results(authors, start_time)

#added for quicker testing of script.py, by not having to run authorSelect
if __name__ == "__main__":
    try:
        f = pd.read_csv("authors.tsv", delimiter='\t', header=None, names=['ID', 'author', 'NA', 'sentence'])
    except Exception as error:
        print(error)
        sys.exit()

    authors = list(set([x["author"] for index, x in f.iterrows()]))
    res = []
    for author in authors:
        i = 0
        scriptRes = main(author)
        res.append(1 if scriptRes[0] == author[1:] else 0)
        print("result for: " + author, scriptRes[0] == author[1:])
        print("time: ", scriptRes[-1])
        i += 1

    print(res)
    print("Average result: ", sum(res)/len(res))