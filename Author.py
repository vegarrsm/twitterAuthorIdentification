import re
from collections import Counter
from math import log, exp
import pandas as pd

class Author:

    #authorID is twitter handle of author (without @)
    #singleTweetID defines whether Author is used in training, on multiple tweets or in 
    #validation on single tweet
    def __init__(self, authorID="", filePath="", singleTweetID = ""):
        self.__authorID = authorID
        self.__wordFrequency = None	#Keeps count of all the words used and their frequency
        self.__wordProbability = {} #Dictionary in the form: {word: (times used)/(total words)}
        self.__tweetLengths = []    #List of length of each tweet
        self.__wordLength = 0       #Average word length for this author
        self.__wordsInText = []	    #All words used in all text that was trained on
        self.__prior = None			#Prior Probability of this Author
        self.__posterior = None		#Posterior Probability of this Author
        self.__filePath = filePath
        self.__singleTweetID = singleTweetID
        self.processFiles()
        

    def getID(self):
        return self.__authorID

    def getWordProbability(self):
        return self.__wordProbability

    def setPrior(self, prior):
        self.__prior = prior

    def getPrior(self):
        return self.__prior

    def setPosterior(self, posterior):
        self.__posterior = posterior

    def getPosterior(self):
        return self.__posterior

    def getWordLength(self):
        return self.__wordLength

    def getTweetLength(self):
        #print("TWEETLENGTH",len(self.__tweetLengths))
        return sum(self.__tweetLengths)/len(self.__tweetLengths) if len(self.__tweetLengths) != 0 else 1

    # Returns conditional probability P(word1|author)
    def getConditional(self, word):
        res = None
        try:
            res = self.__wordProbability[word]
        except:
            res = 0.00000000001 # Sets extremely low value if author has never used the word
        return res


    def processFiles(self):
        f = pd.read_csv(self.__filePath, delimiter='\t', header=None, names=['ID', 'author', 'NA', 'sentence'])
        text = ""
        #Originally implemented it to guess based on all tweets, but quickly realized 
        #i needed more attempts to get useful data
        if self.__singleTweetID == "":    
            for index,row in f.iterrows():
                if row["author"][1:] == self.__authorID:
                    self.__tweetLengths.append(len(row["sentence"]))
                    text += row["sentence"] + " "
        else:
            for index,row in f.iterrows():
                if self.__singleTweetID == index:
                    self.__tweetLengths.append(len(row["sentence"]))
                    text += row["sentence"] + " "
        ## Creates a wordlist from the tweets and appends it to  wordLists
        words = re.sub("[^\w]", " ", text).split()
        self.__wordsInText += words
        self.__wordFrequency = Counter(self.__wordsInText).most_common()
        #print("WORDSLENGTH: ", len(words))
        self.__wordLength = sum(self.__tweetLengths)/len(words) if len(words)!=0 else 1
        ## wordProbability[word] is essentially wordFrequency[word]/wordFrequency.sum(), 
        ## where wordfrequency is how many times a word shows up in the dataset
        for tuple in self.__wordFrequency:
            self.__wordProbability.update({tuple[0]:float(tuple[1])/len(self.__wordsInText)})