import sys
from Author import Author
import time
from math import log, exp

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
def main():
	#testInput = input("Input testdata\n")
	start_time = time.time()
	try:
		f = open("authors.txt", "r", encoding="utf-8")
	except Exception as error:
		print(error)
		sys.exit()

	authors = f.readline().split("@")
	authors[-1] = authors[-1][:-1]
	authors.remove("")
	f.close()
	for i in range(len(authors)):
		authors[i] = Author(authors[i],"authors.txt")
	# Sets prior probability for each author
	for author in authors:
		author.setPrior(1.00/len(authors))
	
	#f = open("testInputFile.txt", "a", encoding="utf-8")
	#f.write("separation---NEW USER---separation\n@testUser\n\n"+testInput+"\nseparation---NEW USER---separation")
	# Creates a testAuthor instance of Author class just for the ease of further processing
	testAuthor = Author("testUser", "testInputFile.txt")
	performNaiveBayes(authors, testAuthor)

	return results(authors, start_time)