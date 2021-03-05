import re
from collections import Counter

class Author:

	# Constructor of the Author class
	#		authorID = could be name or an ID that uniquely identifies the author
	#		filePaths = the list of strings where each string is the filePath to the texts(can be multiple)
	#					written by this specific author. This enables us to train the algorithm in more than
	# 					one text for each author which makes it more diverse

	def __init__(self, authorID="", filePath=""):
		self.__authorID = authorID
		self.__wordFrequency = None							# List of tuples after processing, keeps count of all the words and its frequency
		self.__wordProbability = {}							# This is a dictionary. value is frequency/total words used
		# __wordsInText will contain all the words in all the text used for training the algorithm
		self.__wordsInText = []								# List of all the words from all the text that the training was performed on
		self.__prior = None									# Prior Probability of this Author
		self.__posterior = None								# Posterior Probability of this Author
		self.__filePath = filePath
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

	# Returns conditional probability such as P(word1|author), for this Author
	# If the author has never used the word argument, returns a very small number (heuristic)
	def getConditional(self, word):
		result = None
		try:
			result = self.__wordProbability[word]
		except:
			result = 0.00000000001 # Sets extremely low value if word has never been used
		return result


	def processFiles(self):
		p = re.compile("(?<=separation---NEW USER---separation\n)@"+self.__authorID+"[\s\S]*?(?=separation---NEW USER---separation)")
		f = open(self.__filePath, "r", encoding="utf-8")
		# Reads the entire file as one string, converts to lowercase, replaces \n with a space
		text = p.search(f.read()).group().lower().replace("\n", " ")
		f.close()
		## Creates a wordlist from the tweets and appends it to  wordLists
		words = re.sub("[^\w]", " ", text).split()
		self.__wordsInText += words
		self.__wordFrequency = Counter(self.__wordsInText).most_common()
		
		## wordProbability[word] is essentially wordFrequency[word]/wordFrequency.sum(), where wordfrequency is how many times the word
		## shows up in the dataset
		for tuple in self.__wordFrequency:
			self.__wordProbability.update({tuple[0]:float(tuple[1])/len(self.__wordsInText)})
